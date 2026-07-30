"""Application service for user-owned Agent Skill libraries."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from cywl_oopz.core.observability import opaque_ref

from .availability import MAX_SKILL_DISCOVERY_CHARACTERS
from .errors import AgentSkillLibraryError
from .models import (
    AgentSkill,
    AgentSkillDiscovery,
    AgentSkillInspection,
    AgentSkillInviteResult,
    AgentSkillLibrary,
    AgentSkillResource,
    AgentSkillRevokeResult,
    AgentSkillShareSummary,
    SkillAccessKind,
    SkillOwnershipKind,
    SkillResourceKind,
    SkillShareStatus,
)
from .ports import AgentSkillLibraryStore, SkillShareNotifier

logger = logging.getLogger(__name__)


class AgentSkillLibraryService:
    """Validate content and apply owner-scoped Skill library operations."""

    def __init__(
        self,
        repository: AgentSkillLibraryStore,
        *,
        registered_tools: frozenset[str],
        max_personal_skills: int,
        max_available_skills: int,
        max_resources_per_skill: int,
        max_instruction_characters: int,
        max_resource_characters: int,
        max_accepted_shared_skills: int = 8,
        max_share_recipients_per_call: int = 5,
        notifier: SkillShareNotifier | None = None,
    ) -> None:
        limits = (
            max_personal_skills,
            max_available_skills,
            max_resources_per_skill,
            max_instruction_characters,
            max_resource_characters,
            max_accepted_shared_skills,
            max_share_recipients_per_call,
        )
        if any(limit <= 0 for limit in limits):
            raise ValueError("Agent Skill library limits must be positive")
        self._repository = repository
        self._registered_tools = registered_tools
        self._max_personal_skills = max_personal_skills
        self._max_available_skills = max_available_skills
        self._max_resources_per_skill = max_resources_per_skill
        self._max_instruction_characters = max_instruction_characters
        self._max_resource_characters = max_resource_characters
        self._max_accepted_shared_skills = max_accepted_shared_skills
        self._max_share_recipients_per_call = max_share_recipients_per_call
        self._notifier = notifier

    async def library(self, person_id: str) -> AgentSkillLibrary:
        """Return compact active and archived caller-relative library groups."""
        owned = await self._repository.list_owned(person_id)
        accessible = await self._repository.list_accessible(person_id)
        pending = await self._repository.pending_invitations(person_id)
        outgoing = await self._repository.outgoing_shares(person_id)
        return AgentSkillLibrary(
            owned=owned,
            builtin=tuple(skill for skill in accessible if skill.access is SkillAccessKind.BUILTIN),
            shared=tuple(skill for skill in accessible if skill.access is SkillAccessKind.SHARED),
            pending_invitations=pending,
            outgoing_shares=outgoing,
        )

    async def inspect(
        self,
        person_id: str,
        skill_id: UUID,
        resource_key: str | None = None,
    ) -> AgentSkillInspection:
        """Read fresh instructions and at most one explicitly requested resource."""
        caller = _person_id(person_id)
        inspection = await self._repository.inspect_accessible(caller, skill_id)
        if inspection is None:
            raise AgentSkillLibraryError("skill_not_found")
        if resource_key is None:
            return inspection
        resource = await self._repository.read_inspectable_resource(
            caller,
            skill_id,
            resource_key.strip(),
        )
        if resource is None:
            raise AgentSkillLibraryError("skill_resource_not_found")
        return AgentSkillInspection(
            inspection.bundle,
            inspection.active,
            resource,
        )

    async def create(
        self,
        person_id: str,
        *,
        name: str,
        display_name: str,
        description: str,
        instructions: str,
        required_tools: frozenset[str],
        version: str = "1",
    ) -> AgentSkill:
        """Create one active personal Skill with no resources."""
        owner = _person_id(person_id)
        if len(await self._repository.list_owned(owner)) >= self._max_personal_skills:
            raise AgentSkillLibraryError("skill_library_limit")
        await self._ensure_active_capacity(
            owner,
            extra_count=1,
            description_delta=len(description.strip()),
        )
        self._validate_tools(required_tools)
        self._validate_instructions(instructions)
        try:
            skill = AgentSkill(
                id=uuid4(),
                name=name,
                display_name=display_name,
                description=description,
                instructions=instructions,
                version=version,
                revision=1,
                required_tools=required_tools,
                resources=(),
                metadata={},
                ownership_kind=SkillOwnershipKind.PERSONAL,
                owner_person_id=owner,
            )
        except ValueError as exc:
            raise AgentSkillLibraryError("invalid_agent_skill") from exc
        await self._repository.add_personal(skill)
        logger.info(
            "Personal Agent Skill created: owner=%s skill=%s instructions=%s tools=%s",
            opaque_ref(owner),
            skill.id,
            len(skill.instructions),
            len(skill.required_tools),
        )
        return skill

    async def update(
        self,
        person_id: str,
        skill_id: UUID,
        expected_revision: int,
        *,
        name: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        version: str | None = None,
        required_tools: frozenset[str] | None = None,
    ) -> AgentSkill:
        """Replace selected core fields after a fresh owner revision check."""
        current = await self._owned(person_id, skill_id, expected_revision)
        if required_tools is not None:
            self._validate_tools(required_tools)
        if instructions is not None:
            self._validate_instructions(instructions)
        changes = {
            key: value
            for key, value in {
                "name": name,
                "display_name": display_name,
                "description": description,
                "instructions": instructions,
                "version": version,
                "required_tools": required_tools,
            }.items()
            if value is not None
        }
        if not changes:
            raise AgentSkillLibraryError("skill_no_changes")
        try:
            updated = replace(current, **changes)
        except ValueError as exc:
            raise AgentSkillLibraryError("invalid_agent_skill") from exc
        if updated == current:
            raise AgentSkillLibraryError("skill_no_changes")
        if current.enabled and current.archived_at is None:
            await self._ensure_active_capacity(
                current.owner_person_id or "",
                description_delta=len(updated.description) - len(current.description),
            )
        result = await self._repository.update_owned(updated, expected_revision)
        logger.info(
            "Personal Agent Skill updated: owner=%s skill=%s revision=%s fields=%s",
            opaque_ref(person_id),
            skill_id,
            result.revision,
            ",".join(sorted(changes)),
        )
        return result

    async def upsert_resource(
        self,
        person_id: str,
        skill_id: UUID,
        expected_revision: int,
        *,
        key: str,
        display_name: str,
        description: str,
        kind: SkillResourceKind,
        media_type: str,
        content: str,
        position: int,
    ) -> AgentSkill:
        """Insert or replace one bounded resource by key."""
        current = await self._owned(person_id, skill_id, expected_revision)
        normalized_key = key.strip()
        existing = next(
            (item for item in current.resources if item.key == normalized_key),
            None,
        )
        if existing is None and len(current.resources) >= self._max_resources_per_skill:
            raise AgentSkillLibraryError("skill_resource_library_limit")
        if len(content) > self._max_resource_characters:
            raise AgentSkillLibraryError("skill_resource_too_long")
        try:
            resource = AgentSkillResource(
                id=existing.id if existing is not None else uuid4(),
                key=normalized_key,
                display_name=display_name,
                description=description,
                kind=kind,
                media_type=media_type,
                content=content,
                position=position,
            )
        except ValueError as exc:
            raise AgentSkillLibraryError("invalid_agent_skill_resource") from exc
        if existing == resource:
            raise AgentSkillLibraryError("skill_no_changes")
        return await self._repository.upsert_owned_resource(
            current.owner_person_id or "",
            current.id,
            expected_revision,
            resource,
        )

    async def remove_resource(
        self,
        person_id: str,
        skill_id: UUID,
        expected_revision: int,
        resource_key: str,
    ) -> AgentSkill:
        """Remove one owned resource by its stable key."""
        current = await self._owned(person_id, skill_id, expected_revision)
        key = resource_key.strip()
        if not any(resource.key == key for resource in current.resources):
            raise AgentSkillLibraryError("skill_resource_not_found")
        return await self._repository.remove_owned_resource(
            current.owner_person_id or "",
            skill_id,
            expected_revision,
            key,
        )

    async def set_state(
        self,
        person_id: str,
        skill_id: UUID,
        expected_revision: int,
        *,
        active: bool,
    ) -> AgentSkill:
        """Archive or restore one personal Skill."""
        current = await self._owned(person_id, skill_id, expected_revision)
        if current.enabled is active and (current.archived_at is None) is active:
            return current
        if active:
            await self._ensure_active_capacity(
                current.owner_person_id or "",
                extra_count=1,
                description_delta=len(current.description),
            )
        return await self._repository.set_owned_state(
            current.owner_person_id or "",
            skill_id,
            expected_revision,
            enabled=active,
            archived_at=None if active else datetime.now(UTC),
        )

    async def invite(
        self,
        person_id: str,
        skill_id: UUID,
        mentioned_person_ids: tuple[str, ...],
    ) -> AgentSkillInviteResult:
        """Invite only trusted current-message mention targets."""
        owner = _person_id(person_id)
        current = await self._repository.get_owned(owner, skill_id)
        if current is None:
            raise AgentSkillLibraryError("skill_not_owned")
        if not current.enabled or current.archived_at is not None:
            raise AgentSkillLibraryError("skill_archived")
        recipients = tuple(
            dict.fromkeys(
                recipient.strip()
                for recipient in mentioned_person_ids
                if recipient.strip() and recipient.strip() != owner
            )
        )
        if not recipients:
            raise AgentSkillLibraryError("skill_share_target_required")
        if len(recipients) > self._max_share_recipients_per_call:
            raise AgentSkillLibraryError("skill_share_target_limit")
        now = datetime.now(UTC)
        shares = await self._repository.invite_many(
            owner,
            skill_id,
            recipients,
            now,
        )
        discovery = _owned_discovery(current)
        notification_failures = 0
        for share in shares:
            if not await self._notify_invitation(
                share.recipient_person_id,
                discovery,
            ):
                notification_failures += 1
        return AgentSkillInviteResult(
            discovery,
            shares,
            notification_failures,
        )

    async def respond(
        self,
        person_id: str,
        share_id: UUID,
        *,
        accepted: bool,
    ) -> AgentSkillShareSummary:
        """Accept or decline one caller-owned invitation."""
        recipient = _person_id(person_id)
        current = await self._repository.share_for_recipient(recipient, share_id)
        if current is None:
            raise AgentSkillLibraryError("skill_invitation_not_found")
        target_status = SkillShareStatus.ACCEPTED if accepted else SkillShareStatus.DECLINED
        if (
            current.share.status is not SkillShareStatus.PENDING
            and current.share.status is not target_status
        ):
            raise AgentSkillLibraryError("skill_invitation_answered")
        if accepted and current.share.status is not SkillShareStatus.ACCEPTED:
            library = await self.library(recipient)
            if len(library.shared) >= self._max_accepted_shared_skills:
                raise AgentSkillLibraryError("skill_shared_library_limit")
            if current.active:
                await self._ensure_active_capacity(
                    recipient,
                    extra_count=1,
                    description_delta=len(current.skill.description),
                )
        await self._repository.respond(
            recipient,
            share_id,
            target_status,
            datetime.now(UTC),
        )
        updated = await self._repository.share_for_recipient(recipient, share_id)
        if updated is None:
            raise AgentSkillLibraryError("skill_invitation_not_found")
        return updated

    async def revoke(
        self,
        person_id: str,
        skill_id: UUID,
        mentioned_person_ids: tuple[str, ...],
        *,
        revoke_all: bool,
    ) -> AgentSkillRevokeResult:
        """Revoke mentioned recipients or every grant for one owned Skill."""
        owner = _person_id(person_id)
        current = await self._repository.get_owned(owner, skill_id)
        if current is None:
            raise AgentSkillLibraryError("skill_not_owned")
        recipients = tuple(
            dict.fromkeys(
                recipient.strip()
                for recipient in mentioned_person_ids
                if recipient.strip() and recipient.strip() != owner
            )
        )
        if revoke_all and recipients:
            raise AgentSkillLibraryError("skill_share_target_conflict")
        if not revoke_all and not recipients:
            raise AgentSkillLibraryError("skill_share_target_required")
        removed = await self._repository.revoke_owned_shares(
            owner,
            skill_id,
            None if revoke_all else recipients,
        )
        if not removed:
            raise AgentSkillLibraryError("skill_share_not_found")
        discovery = _owned_discovery(current)
        notification_failures = 0
        for share in removed:
            if not await self._notify_revoked(
                share.recipient_person_id,
                discovery,
            ):
                notification_failures += 1
        return AgentSkillRevokeResult(
            discovery,
            removed,
            notification_failures,
        )

    async def _owned(
        self,
        person_id: str,
        skill_id: UUID,
        expected_revision: int,
    ) -> AgentSkill:
        current = await self._repository.get_owned(_person_id(person_id), skill_id)
        if current is None:
            raise AgentSkillLibraryError("skill_not_owned")
        if current.revision != expected_revision:
            raise AgentSkillLibraryError("skill_revision_conflict")
        return current

    def _validate_tools(self, required_tools: frozenset[str]) -> None:
        if not required_tools.issubset(self._registered_tools):
            raise AgentSkillLibraryError("skill_unknown_required_tools")

    def _validate_instructions(self, instructions: str) -> None:
        if len(instructions) > self._max_instruction_characters:
            raise AgentSkillLibraryError("skill_instructions_too_long")

    async def _ensure_active_capacity(
        self,
        person_id: str,
        *,
        extra_count: int = 0,
        description_delta: int = 0,
    ) -> None:
        accessible = await self._repository.list_accessible(person_id)
        if len(accessible) + extra_count > self._max_available_skills:
            raise AgentSkillLibraryError("skill_library_limit")
        description_characters = sum(len(skill.description) for skill in accessible)
        if description_characters + description_delta > MAX_SKILL_DISCOVERY_CHARACTERS:
            raise AgentSkillLibraryError("skill_library_limit")

    async def _notify_invitation(
        self,
        recipient_person_id: str,
        skill: AgentSkillDiscovery,
    ) -> bool:
        if self._notifier is None:
            return True
        try:
            return await self._notifier.invitation(recipient_person_id, skill)
        except Exception:
            logger.warning(
                "Skill invitation notification failed: recipient=%s skill=%s",
                opaque_ref(recipient_person_id),
                skill.id,
            )
            return False

    async def _notify_revoked(
        self,
        recipient_person_id: str,
        skill: AgentSkillDiscovery,
    ) -> bool:
        if self._notifier is None:
            return True
        try:
            return await self._notifier.revoked(recipient_person_id, skill)
        except Exception:
            logger.warning(
                "Skill revoke notification failed: recipient=%s skill=%s",
                opaque_ref(recipient_person_id),
                skill.id,
            )
            return False


def _person_id(value: str) -> str:
    person_id = value.strip()
    if not person_id:
        raise ValueError("Skill library person ID must not be empty")
    return person_id


def _owned_discovery(skill: AgentSkill) -> AgentSkillDiscovery:
    return AgentSkillDiscovery(
        id=skill.id,
        name=skill.name,
        display_name=skill.display_name,
        description=skill.description,
        version=skill.version,
        revision=skill.revision,
        required_tools=skill.required_tools,
        access=SkillAccessKind.OWNED,
    )
