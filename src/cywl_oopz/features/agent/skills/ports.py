"""Persistence ports for Agent skill bundles."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from .models import AgentSkill, AgentSkillShare, SkillShareStatus


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

    async def invite(
        self,
        owner_person_id: str,
        skill_id: UUID,
        recipient_person_id: str,
        now: datetime,
    ) -> AgentSkillShare:
        """Create or refresh one recipient invitation for an owned Skill."""

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
    ) -> bool:
        """Delete one share belonging to an owned Skill."""
