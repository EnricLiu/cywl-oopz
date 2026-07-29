"""Deterministic Agent skill visibility derived from the actual run tools."""

from __future__ import annotations

import logging

from .catalog import AgentSkillCatalogSnapshot
from .models import AgentSkill

logger = logging.getLogger(__name__)

LOAD_AGENT_SKILL_TOOL = "load_agent_skill"


class SkillAvailabilityService:
    """Filter a pinned catalog without granting any additional tool capability."""

    def resolve(
        self,
        snapshot: AgentSkillCatalogSnapshot,
        enabled_tools: tuple[str, ...],
    ) -> tuple[AgentSkill, ...]:
        """Return skills whose complete required tool set is visible this run."""
        enabled = frozenset(enabled_tools)
        if not snapshot.loaded or LOAD_AGENT_SKILL_TOOL not in enabled:
            return ()
        available = tuple(
            skill for skill in snapshot.skills.values() if skill.required_tools.issubset(enabled)
        )
        logger.debug(
            "Resolved Agent skills: catalog=%s available=%s filtered=%s",
            len(snapshot.skills),
            len(available),
            len(snapshot.skills) - len(available),
        )
        return available
