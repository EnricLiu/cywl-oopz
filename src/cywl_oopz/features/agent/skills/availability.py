"""Deterministic Agent skill visibility derived from the actual run tools."""

from __future__ import annotations

import logging

from .models import AgentSkillDiscovery

logger = logging.getLogger(__name__)

LOAD_AGENT_SKILL_TOOL = "load_agent_skill"
MAX_SKILL_DISCOVERY_CHARACTERS = 8_000


class SkillAvailabilityCapacityError(ValueError):
    """The caller-visible discovery set exceeds a configured run budget."""


class SkillAvailabilityService:
    """Filter a pinned catalog without granting any additional tool capability."""

    def __init__(self, max_available_skills: int = 32) -> None:
        if max_available_skills <= 0:
            raise ValueError("Maximum available Skills must be positive")
        self._max_available_skills = max_available_skills

    def resolve(
        self,
        discoveries: tuple[AgentSkillDiscovery, ...],
        enabled_tools: tuple[str, ...],
    ) -> tuple[AgentSkillDiscovery, ...]:
        """Return discoveries whose complete required tool set is visible this run."""
        enabled = frozenset(enabled_tools)
        if LOAD_AGENT_SKILL_TOOL not in enabled:
            return ()
        available = tuple(skill for skill in discoveries if skill.required_tools.issubset(enabled))
        if len(available) > self._max_available_skills:
            raise SkillAvailabilityCapacityError(
                "Agent Skill library exceeds the configured Skill count"
            )
        if sum(len(skill.description) for skill in available) > MAX_SKILL_DISCOVERY_CHARACTERS:
            raise SkillAvailabilityCapacityError(
                "Agent Skill library exceeds the description character budget"
            )
        logger.debug(
            "Resolved Agent skills: catalog=%s available=%s filtered=%s",
            len(discoveries),
            len(available),
            len(discoveries) - len(available),
        )
        return available
