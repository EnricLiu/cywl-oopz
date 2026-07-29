"""Stable domain errors for user-owned Agent Skill persistence."""

from __future__ import annotations


class AgentSkillConflictError(ValueError):
    """A Skill or share uniqueness constraint conflicts with current state."""


class AgentSkillNotFoundError(ValueError):
    """The caller cannot access the requested Skill or share."""


class AgentSkillRevisionConflictError(ValueError):
    """The requested mutation was based on an obsolete Skill revision."""
