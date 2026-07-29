"""Run-local Agent skill snapshot and atomic progressive-disclosure budgets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID

from .catalog import AgentSkillCatalogSnapshot
from .models import AgentSkill, AgentSkillResource


class AgentSkillScopeError(Exception):
    """Expected scope rejection represented by a stable model-visible code."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


@dataclass(frozen=True, slots=True)
class AgentSkillActivation:
    """Result of loading one Skill instructions document."""

    skill: AgentSkill
    already_loaded: bool
    returned_characters: int


@dataclass(frozen=True, slots=True)
class AgentSkillResourceRead:
    """Result of loading one additional Skill text resource."""

    skill: AgentSkill
    resource: AgentSkillResource
    already_loaded: bool
    returned_characters: int


class AgentSkillRunScope:
    """Pin visible skills and enforce all per-run disclosure limits atomically."""

    def __init__(
        self,
        snapshot: AgentSkillCatalogSnapshot,
        available_skills: tuple[AgentSkill, ...],
        *,
        max_activations: int,
        max_resources: int,
        max_instruction_characters: int,
        max_resource_characters: int,
        max_context_characters: int,
    ) -> None:
        limits = (
            max_activations,
            max_resources,
            max_instruction_characters,
            max_resource_characters,
            max_context_characters,
        )
        if any(value <= 0 for value in limits):
            raise ValueError("Agent skill run limits must be positive")
        if not snapshot.loaded:
            raise ValueError("Agent skill run scope requires a loaded catalog snapshot")
        available = {skill.name: skill for skill in available_skills}
        if len(available) != len(available_skills):
            raise ValueError("Agent skill run scope contains duplicate names")
        if any(snapshot.skills.get(name) is not skill for name, skill in available.items()):
            raise ValueError("Available skills must belong to the pinned catalog snapshot")

        self._catalog = snapshot.skills
        self._available = MappingProxyType(available)
        self._max_activations = max_activations
        self._max_resources = max_resources
        self._max_instruction_characters = max_instruction_characters
        self._max_resource_characters = max_resource_characters
        self._max_context_characters = max_context_characters
        self._activated: set[str] = set()
        self._read_resources: set[tuple[str, UUID]] = set()
        self._returned_characters = 0
        self._lock = asyncio.Lock()

    @property
    def available_skills(self) -> tuple[AgentSkill, ...]:
        """Return the run-pinned visible Skills in stable catalog order."""
        return tuple(self._available.values())

    @property
    def activation_count(self) -> int:
        return len(self._activated)

    @property
    def resource_count(self) -> int:
        return len(self._read_resources)

    @property
    def returned_characters(self) -> int:
        return self._returned_characters

    async def load(self, name: str) -> AgentSkillActivation:
        """Load instructions once, charging distinct activations under one lock."""
        async with self._lock:
            skill = self._resolve(name)
            if skill.name in self._activated:
                return AgentSkillActivation(skill, True, 0)
            characters = len(skill.instructions)
            if len(self._activated) >= self._max_activations:
                raise AgentSkillScopeError("skill_activation_limit")
            self._check_characters(
                characters,
                item_limit=self._max_instruction_characters,
            )
            self._activated.add(skill.name)
            self._returned_characters += characters
            return AgentSkillActivation(skill, False, characters)

    async def read_resource(
        self,
        skill_name: str,
        resource_id: UUID,
    ) -> AgentSkillResourceRead:
        """Read an activated Skill resource once under the shared context budget."""
        async with self._lock:
            skill = self._resolve(skill_name)
            if skill.name not in self._activated:
                raise AgentSkillScopeError("skill_not_activated")
            resource = next(
                (item for item in skill.resources if item.id == resource_id),
                None,
            )
            if resource is None:
                raise AgentSkillScopeError("skill_resource_not_found")
            key = (skill.name, resource.id)
            if key in self._read_resources:
                return AgentSkillResourceRead(skill, resource, True, 0)
            if len(self._read_resources) >= self._max_resources:
                raise AgentSkillScopeError("skill_resource_limit")
            characters = len(resource.content)
            self._check_characters(
                characters,
                item_limit=self._max_resource_characters,
            )
            self._read_resources.add(key)
            self._returned_characters += characters
            return AgentSkillResourceRead(skill, resource, False, characters)

    def _resolve(self, name: str) -> AgentSkill:
        skill = self._catalog.get(name)
        if skill is None:
            raise AgentSkillScopeError("skill_not_found")
        available = self._available.get(name)
        if available is None:
            raise AgentSkillScopeError("skill_not_available")
        return available

    def _check_characters(self, characters: int, *, item_limit: int) -> None:
        if (
            characters > item_limit
            or self._returned_characters + characters > self._max_context_characters
        ):
            raise AgentSkillScopeError("skill_context_limit")
