"""Atomically reloadable in-memory catalog for PostgreSQL Agent skills."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID

from cywl_oopz.core.health import HealthRegistry, HealthState
from cywl_oopz.core.observability import exception_kind

from .models import AgentSkill
from .ports import AgentSkillRepository

logger = logging.getLogger(__name__)

MAX_CATALOG_DESCRIPTION_CHARACTERS = 8_000


class AgentSkillCatalogCapacityError(ValueError):
    """The complete valid catalog cannot fit the configured discovery budget."""


@dataclass(frozen=True, slots=True)
class AgentSkillCatalogDiagnostic:
    """One safe maintenance diagnostic without skill instructions or resources."""

    skill_name: str
    code: str
    names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentSkillCatalogSnapshot:
    """Immutable generation-pinned catalog published to Agent runs."""

    generation: int
    skills: Mapping[str, AgentSkill]
    diagnostics: tuple[AgentSkillCatalogDiagnostic, ...] = ()
    loaded: bool = True

    @classmethod
    def empty(cls) -> AgentSkillCatalogSnapshot:
        """Return the startup state before PostgreSQL has loaded successfully."""
        return cls(0, MappingProxyType({}), (), False)

    @classmethod
    def build(
        cls,
        skills: Iterable[AgentSkill],
        *,
        generation: int,
        registered_tools: frozenset[str],
        max_available_skills: int,
    ) -> AgentSkillCatalogSnapshot:
        """Validate dependencies and build a deterministic all-or-nothing snapshot."""
        if generation <= 0:
            raise ValueError("Agent skill catalog generation must be positive")
        if max_available_skills <= 0:
            raise ValueError("Maximum available skills must be positive")

        values = tuple(skills)
        ids: set[UUID] = set()
        names: set[str] = set()
        accepted: list[AgentSkill] = []
        diagnostics: list[AgentSkillCatalogDiagnostic] = []
        for skill in values:
            if skill.id in ids:
                raise ValueError("Agent skill catalog contains duplicate IDs")
            if skill.name in names:
                raise ValueError("Agent skill catalog contains duplicate names")
            ids.add(skill.id)
            names.add(skill.name)
            unknown = tuple(sorted(skill.required_tools.difference(registered_tools)))
            if unknown:
                diagnostics.append(
                    AgentSkillCatalogDiagnostic(
                        skill.name,
                        "unknown_required_tools",
                        unknown,
                    )
                )
                continue
            accepted.append(skill)

        if len(accepted) > max_available_skills:
            raise AgentSkillCatalogCapacityError(
                "Agent skill catalog exceeds the configured skill count"
            )
        description_characters = sum(len(skill.description) for skill in accepted)
        if description_characters > MAX_CATALOG_DESCRIPTION_CHARACTERS:
            raise AgentSkillCatalogCapacityError(
                "Agent skill catalog exceeds the description character budget"
            )

        ordered = sorted(accepted, key=lambda item: item.name)
        return cls(
            generation,
            MappingProxyType({skill.name: skill for skill in ordered}),
            tuple(diagnostics),
        )


class ReloadableAgentSkillCatalog:
    """Refresh immutable snapshots by cheap generation checks on run boundaries."""

    def __init__(
        self,
        repository: AgentSkillRepository,
        *,
        registered_tools: tuple[str, ...],
        refresh_seconds: float,
        max_available_skills: int,
        health: HealthRegistry | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if refresh_seconds <= 0:
            raise ValueError("Agent skill catalog refresh interval must be positive")
        self._repository = repository
        self._registered_tools = frozenset(registered_tools)
        self._refresh_seconds = refresh_seconds
        self._max_available_skills = max_available_skills
        self._health = health
        self._clock = clock
        self._snapshot = AgentSkillCatalogSnapshot.empty()
        self._next_refresh_at = 0.0
        self._reload_lock = asyncio.Lock()

    @property
    def snapshot(self) -> AgentSkillCatalogSnapshot:
        """Return the current immutable snapshot without acquiring a lock."""
        return self._snapshot

    async def start(self) -> None:
        """Try the initial load while allowing ordinary Agent chat to degrade cleanly."""
        try:
            await self.reload()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Initial Agent skill catalog load failed: error=%s",
                exception_kind(exc),
            )

    async def refresh_if_stale(self) -> bool:
        """Refresh after the TTL and return whether a new snapshot was published."""
        now = self._clock()
        if now < self._next_refresh_at:
            return False
        async with self._reload_lock:
            now = self._clock()
            if now < self._next_refresh_at:
                return False
            try:
                generation = await self._repository.generation()
                if self._snapshot.loaded and generation == self._snapshot.generation:
                    self._next_refresh_at = now + self._refresh_seconds
                    self._mark_snapshot_health(self._snapshot)
                    return False
                await self._reload_locked(generation, now)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._next_refresh_at = now + self._refresh_seconds
                self._mark_failed_health()
                logger.warning(
                    "Agent skill catalog refresh failed; retaining previous snapshot: "
                    "generation=%s error=%s",
                    self._snapshot.generation,
                    exception_kind(exc),
                )
                return False

    async def reload(self) -> AgentSkillCatalogSnapshot:
        """Force a full load, preserving the previous snapshot if validation fails."""
        async with self._reload_lock:
            now = self._clock()
            try:
                generation = await self._repository.generation()
                return await self._reload_locked(generation, now)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._next_refresh_at = now + self._refresh_seconds
                self._mark_failed_health()
                raise

    async def _reload_locked(
        self,
        generation: int,
        now: float,
    ) -> AgentSkillCatalogSnapshot:
        started_at = time.perf_counter()
        skills = await self._repository.load_enabled()
        replacement = AgentSkillCatalogSnapshot.build(
            skills,
            generation=generation,
            registered_tools=self._registered_tools,
            max_available_skills=self._max_available_skills,
        )
        self._snapshot = replacement
        self._next_refresh_at = now + self._refresh_seconds
        self._mark_snapshot_health(replacement)
        for diagnostic in replacement.diagnostics:
            logger.warning(
                "Agent skill skipped during catalog reload: skill=%s code=%s names=%s",
                diagnostic.skill_name,
                diagnostic.code,
                ",".join(diagnostic.names),
            )
        logger.info(
            "Agent skill catalog reloaded: generation=%s skills=%s diagnostics=%s "
            "elapsed_seconds=%.3f",
            replacement.generation,
            len(replacement.skills),
            len(replacement.diagnostics),
            time.perf_counter() - started_at,
        )
        return replacement

    def _mark_snapshot_health(self, snapshot: AgentSkillCatalogSnapshot) -> None:
        if self._health is None:
            return
        if snapshot.diagnostics:
            self._health.mark("skills", HealthState.DEGRADED, "invalid skills skipped")
        else:
            self._health.mark("skills", HealthState.HEALTHY, "catalog loaded")

    def _mark_failed_health(self) -> None:
        if self._health is not None:
            self._health.mark("skills", HealthState.DEGRADED, "catalog refresh failed")
