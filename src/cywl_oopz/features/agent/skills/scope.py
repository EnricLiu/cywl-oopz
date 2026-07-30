"""Run-local user-scoped Agent Skill discovery and disclosure budgets."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID

from .errors import AgentSkillRevisionConflictError
from .models import (
    AgentSkillBundle,
    AgentSkillDiscovery,
    AgentSkillResource,
)
from .ports import AgentSkillReadRepository


class AgentSkillScopeError(Exception):
    """Expected scope rejection represented by a stable model-visible code."""

    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


@dataclass(frozen=True, slots=True)
class AgentSkillActivation:
    """Result of loading one Skill instructions document."""

    bundle: AgentSkillBundle
    already_loaded: bool
    returned_characters: int


@dataclass(frozen=True, slots=True)
class AgentSkillResourceRead:
    """Result of loading one additional Skill text resource."""

    discovery: AgentSkillDiscovery
    resource: AgentSkillResource
    already_loaded: bool
    returned_characters: int


class AgentSkillRunScope:
    """Pin caller-visible discovery and enforce progressive-disclosure limits."""

    def __init__(
        self,
        repository: AgentSkillReadRepository,
        person_id: str,
        available_skills: tuple[AgentSkillDiscovery, ...],
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
            raise ValueError("Agent Skill run limits must be positive")
        caller = person_id.strip()
        if not caller:
            raise ValueError("Agent Skill run person ID must not be empty")
        available = {skill.id: skill for skill in available_skills}
        if len(available) != len(available_skills):
            raise ValueError("Agent Skill run scope contains duplicate IDs")
        self._repository = repository
        self._person_id = caller
        self._available = MappingProxyType(available)
        self._max_activations = max_activations
        self._max_resources = max_resources
        self._max_instruction_characters = max_instruction_characters
        self._max_resource_characters = max_resource_characters
        self._max_context_characters = max_context_characters
        self._activated: set[UUID] = set()
        self._read_resources: set[tuple[UUID, UUID]] = set()
        self._returned_characters = 0
        self._lock = asyncio.Lock()

    @property
    def available_skills(self) -> tuple[AgentSkillDiscovery, ...]:
        """Return run-pinned discovery in repository order."""
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

    async def load(self, skill_id: UUID) -> AgentSkillActivation:
        """Load instructions at the run-pinned revision and charge them once."""
        discovery = self._resolve(skill_id)
        try:
            bundle = await self._repository.load_accessible_bundle(
                self._person_id,
                skill_id,
                discovery.revision,
            )
        except AgentSkillRevisionConflictError as exc:
            raise AgentSkillScopeError("skill_revision_changed") from exc
        if bundle is None:
            raise AgentSkillScopeError("skill_not_available")
        if bundle.discovery.id != discovery.id or bundle.discovery.revision != discovery.revision:
            raise AgentSkillScopeError("skill_revision_changed")

        async with self._lock:
            self._resolve(skill_id)
            if skill_id in self._activated:
                return AgentSkillActivation(bundle, True, 0)
            characters = len(bundle.instructions)
            if len(self._activated) >= self._max_activations:
                raise AgentSkillScopeError("skill_activation_limit")
            self._check_characters(
                characters,
                item_limit=self._max_instruction_characters,
            )
            self._activated.add(skill_id)
            self._returned_characters += characters
            return AgentSkillActivation(bundle, False, characters)

    async def read_resource(
        self,
        skill_id: UUID,
        resource_id: UUID,
    ) -> AgentSkillResourceRead:
        """Read one activated Skill resource at the run-pinned revision."""
        discovery = self._resolve(skill_id)
        async with self._lock:
            if skill_id not in self._activated:
                raise AgentSkillScopeError("skill_not_activated")
        try:
            resource = await self._repository.read_accessible_resource(
                self._person_id,
                skill_id,
                resource_id,
                discovery.revision,
            )
        except AgentSkillRevisionConflictError as exc:
            raise AgentSkillScopeError("skill_revision_changed") from exc
        if resource is None:
            raise AgentSkillScopeError("skill_resource_not_found")

        async with self._lock:
            self._resolve(skill_id)
            if skill_id not in self._activated:
                raise AgentSkillScopeError("skill_not_activated")
            key = (skill_id, resource.id)
            if key in self._read_resources:
                return AgentSkillResourceRead(discovery, resource, True, 0)
            if len(self._read_resources) >= self._max_resources:
                raise AgentSkillScopeError("skill_resource_limit")
            characters = len(resource.content)
            self._check_characters(
                characters,
                item_limit=self._max_resource_characters,
            )
            self._read_resources.add(key)
            self._returned_characters += characters
            return AgentSkillResourceRead(discovery, resource, False, characters)

    def _resolve(self, skill_id: UUID) -> AgentSkillDiscovery:
        skill = self._available.get(skill_id)
        if skill is None:
            raise AgentSkillScopeError("skill_not_available")
        return skill

    def _check_characters(self, characters: int, *, item_limit: int) -> None:
        if (
            characters > item_limit
            or self._returned_characters + characters > self._max_context_characters
        ):
            raise AgentSkillScopeError("skill_context_limit")
