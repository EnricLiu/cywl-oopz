"""Persistence ports for Agent skill bundles."""

from __future__ import annotations

from typing import Protocol

from .models import AgentSkill


class AgentSkillRepository(Protocol):
    """Read enabled skill bundles and their catalog generation."""

    async def load_enabled(self) -> tuple[AgentSkill, ...]:
        """Load all enabled skills with their ordered text resources."""

    async def generation(self) -> int:
        """Return the monotonically increasing catalog generation."""
