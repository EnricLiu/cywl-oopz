"""Persistence ports for Agent skill bundles."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from .models import (
    AgentSkill,
    AgentSkillBundle,
    AgentSkillDiscovery,
    AgentSkillInspection,
    AgentSkillOutgoingShare,
    AgentSkillOwnedSummary,
    AgentSkillResource,
    AgentSkillShare,
    AgentSkillShareSummary,
    SkillShareStatus,
)


class AgentSkillRepository(Protocol):
    """Read enabled skill bundles and their catalog generation."""

    async def load_enabled(self) -> tuple[AgentSkill, ...]:
        """Load all enabled skills with their ordered text resources."""

    async def generation(self) -> int:
        """Return the monotonically increasing catalog generation."""


class AgentSkillLibraryRepository(Protocol):
    """Owner- and recipient-scoped persistence for personal Skill libraries."""

    async def load_accessible(self, person_id: str) -> tuple[AgentSkill, ...]:
        """Load active builtin, owned, and accepted shared Skill bundles."""

    async def add_personal(self, skill: AgentSkill) -> None:
        """Persist one complete personal Skill bundle."""

    async def get_owned(self, person_id: str, skill_id: UUID) -> AgentSkill | None:
        """Load one Skill only when the caller owns it."""

    async def list_owned(self, person_id: str) -> tuple[AgentSkillOwnedSummary, ...]:
        """List active and archived personal Skill metadata."""

    async def inspect_accessible(
        self,
        person_id: str,
        skill_id: UUID,
    ) -> AgentSkillInspection | None:
        """Read fresh instructions/manifests without resource content."""

    async def read_inspectable_resource(
        self,
        person_id: str,
        skill_id: UUID,
        resource_key: str,
    ) -> AgentSkillResource | None:
        """Read one resource visible to management inspection."""

    async def update_owned(
        self,
        skill: AgentSkill,
        expected_revision: int,
    ) -> AgentSkill:
        """Replace core fields under an owner/revision lock."""

    async def upsert_owned_resource(
        self,
        owner_person_id: str,
        skill_id: UUID,
        expected_revision: int,
        resource: AgentSkillResource,
    ) -> AgentSkill:
        """Insert or replace one resource under an owner/revision lock."""

    async def remove_owned_resource(
        self,
        owner_person_id: str,
        skill_id: UUID,
        expected_revision: int,
        resource_key: str,
    ) -> AgentSkill:
        """Remove one resource under an owner/revision lock."""

    async def set_owned_state(
        self,
        owner_person_id: str,
        skill_id: UUID,
        expected_revision: int,
        *,
        enabled: bool,
        archived_at: datetime | None,
    ) -> AgentSkill:
        """Archive or restore one Skill under an owner/revision lock."""

    async def invite(
        self,
        owner_person_id: str,
        skill_id: UUID,
        recipient_person_id: str,
        now: datetime,
    ) -> AgentSkillShare:
        """Create or refresh one recipient invitation for an owned Skill."""

    async def invite_many(
        self,
        owner_person_id: str,
        skill_id: UUID,
        recipient_person_ids: tuple[str, ...],
        now: datetime,
    ) -> tuple[AgentSkillShare, ...]:
        """Atomically create or refresh several recipient invitations."""

    async def pending_invitations(
        self,
        recipient_person_id: str,
    ) -> tuple[AgentSkillShareSummary, ...]:
        """List pending invitations for one recipient."""

    async def outgoing_shares(
        self,
        owner_person_id: str,
    ) -> tuple[AgentSkillOutgoingShare, ...]:
        """Aggregate pending and accepted shares for one owner."""

    async def share_for_recipient(
        self,
        recipient_person_id: str,
        share_id: UUID,
    ) -> AgentSkillShareSummary | None:
        """Read one recipient-scoped share and safe Skill metadata."""

    async def share_for_owner(
        self,
        owner_person_id: str,
        share_id: UUID,
    ) -> AgentSkillShareSummary | None:
        """Read one owner-scoped share and safe Skill metadata."""

    async def respond(
        self,
        recipient_person_id: str,
        share_id: UUID,
        status: SkillShareStatus,
        now: datetime,
    ) -> AgentSkillShare:
        """Accept or decline one invitation belonging to the recipient."""

    async def revoke(
        self,
        owner_person_id: str,
        share_id: UUID,
    ) -> AgentSkillShare | None:
        """Delete and return one share belonging to an owned Skill."""

    async def revoke_owned_shares(
        self,
        owner_person_id: str,
        skill_id: UUID,
        recipient_person_ids: tuple[str, ...] | None,
    ) -> tuple[AgentSkillShare, ...]:
        """Atomically revoke selected recipients, or every share for one owned Skill."""


class SkillShareNotifier(Protocol):
    """Best-effort private notification boundary for Skill sharing."""

    async def invitation(
        self,
        recipient_person_id: str,
        skill: AgentSkillDiscovery,
    ) -> bool:
        """Notify one recipient about an invitation."""

    async def revoked(
        self,
        recipient_person_id: str,
        skill: AgentSkillDiscovery,
    ) -> bool:
        """Notify one recipient that a grant was revoked."""


class AgentSkillReadRepository(Protocol):
    """Three-level, caller-scoped progressive disclosure reads."""

    async def list_accessible(self, person_id: str) -> tuple[AgentSkillDiscovery, ...]:
        """Return only discovery metadata visible to one caller."""

    async def load_accessible_bundle(
        self,
        person_id: str,
        skill_id: UUID,
        revision: int,
    ) -> AgentSkillBundle | None:
        """Load instructions and resource manifests at one pinned revision."""

    async def read_accessible_resource(
        self,
        person_id: str,
        skill_id: UUID,
        resource_id: UUID,
        revision: int,
    ) -> AgentSkillResource | None:
        """Load one resource body at one pinned revision."""


class AgentSkillLibraryStore(
    AgentSkillLibraryRepository,
    AgentSkillReadRepository,
    Protocol,
):
    """Complete persistence boundary used by the library application service."""
