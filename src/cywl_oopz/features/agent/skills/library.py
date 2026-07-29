"""Application service for user-owned Agent Skill libraries."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from cywl_oopz.core.observability import opaque_ref

from .catalog import MAX_CATALOG_DESCRIPTION_CHARACTERS
from .errors import AgentSkillLibraryError
from .models import (
    AgentSkill,
    AgentSkillInspection,
    AgentSkillLibrary,
    AgentSkillResource,
    SkillAccessKind,
    SkillOwnershipKind,
    SkillResourceKind,
)
from .ports import AgentSkillLibraryStore

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
    ) -> None:
        limits = (
            max_personal_skills,
            max_available_skills,
            max_resources_per_skill,
            max_instruction_characters,
            max_resource_characters,
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

    async def library(self, person_id: str) -> AgentSkillLibrary:
        """Return compact active and archived caller-relative library groups."""
        owned = await self._repository.list_owned(person_id)
        accessible = await self._repository.list_accessible(person_id)
        return AgentSkillLibrary(
            owned=owned,
            builtin=tuple(skill for skill in accessible if skill.access is SkillAccessKind.BUILTIN),
            shared=tuple(skill for skill in accessible if skill.access is SkillAccessKind.SHARED),
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
        if description_characters + description_delta > MAX_CATALOG_DESCRIPTION_CHARACTERS:
            raise AgentSkillLibraryError("skill_library_limit")


def _person_id(value: str) -> str:
    person_id = value.strip()
    if not person_id:
        raise ValueError("Skill library person ID must not be empty")
    return person_id
