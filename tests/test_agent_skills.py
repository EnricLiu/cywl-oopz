from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from uuid import uuid4

import pytest

from cywl_oopz.core.health import HealthRegistry, HealthState
from cywl_oopz.features.agent.skills.availability import SkillAvailabilityService
from cywl_oopz.features.agent.skills.catalog import (
    AgentSkillCatalogCapacityError,
    AgentSkillCatalogSnapshot,
    ReloadableAgentSkillCatalog,
)
from cywl_oopz.features.agent.skills.models import (
    AgentSkill,
    AgentSkillResource,
    SkillResourceKind,
)


def resource(*, position: int = 1) -> AgentSkillResource:
    return AgentSkillResource(
        id=uuid4(),
        key=f"guide-{position}",
        display_name=f"Guide {position}",
        description="Read when the workflow needs more detail.",
        kind=SkillResourceKind.REFERENCE,
        media_type="text/markdown",
        content="# Guide\nFollow the verified steps.",
        position=position,
    )


def skill(*, resources: tuple[AgentSkillResource, ...] = ()) -> AgentSkill:
    return AgentSkill(
        id=uuid4(),
        name="web-research",
        display_name="网页研究",
        description="Search and read primary sources for current factual questions.",
        instructions="Search, read the important pages, then cite the URLs actually used.",
        version="1",
        revision=1,
        required_tools=frozenset({"search_web", "read_web_page"}),
        resources=resources,
        metadata={"tags": ["web", {"mode": "research"}]},
    )


class FakeSkillRepository:
    def __init__(self, skills: tuple[AgentSkill, ...], generation: int = 1) -> None:
        self.skills = skills
        self.generation_value = generation
        self.generation_calls = 0
        self.load_calls = 0
        self.error: Exception | None = None

    async def load_enabled(self) -> tuple[AgentSkill, ...]:
        self.load_calls += 1
        if self.error is not None:
            raise self.error
        return self.skills

    async def generation(self) -> int:
        self.generation_calls += 1
        if self.error is not None:
            raise self.error
        return self.generation_value


def test_skill_domain_normalizes_and_deeply_freezes_catalog_values() -> None:
    second = resource(position=2)
    first = resource(position=1)

    value = skill(resources=(second, first))

    assert [item.position for item in value.resources] == [1, 2]
    assert isinstance(value.metadata, MappingProxyType)
    assert value.metadata["tags"] == ("web", MappingProxyType({"mode": "research"}))
    with pytest.raises(TypeError):
        value.metadata["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "Web Research", "Skill name"),
        ("display_name", "", "display name"),
        ("description", " ", "description"),
        ("instructions", "", "instructions"),
        ("version", "line\nbreak", "version"),
        ("revision", 0, "revision"),
        ("required_tools", frozenset({"not-a-tool"}), "required tools"),
    ],
)
def test_skill_domain_rejects_invalid_bundle_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    values = {
        "id": uuid4(),
        "name": "web-research",
        "display_name": "网页研究",
        "description": "Research current facts.",
        "instructions": "Search and read sources.",
        "version": "1",
        "revision": 1,
        "required_tools": frozenset({"search_web"}),
        "resources": (),
        "metadata": {},
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        AgentSkill(**values)  # type: ignore[arg-type]


def test_skill_domain_rejects_duplicate_resource_identity_and_position() -> None:
    first = resource()
    duplicate_position = replace(resource(position=2), position=1)

    with pytest.raises(ValueError, match="positions"):
        skill(resources=(first, duplicate_position))


def test_skill_catalog_is_stable_immutable_and_skips_unknown_tool_dependencies() -> None:
    accepted = skill()
    skipped = AgentSkill(
        id=uuid4(),
        name="unknown-tools",
        display_name="Unknown",
        description="References an unavailable tool.",
        instructions="Use the unavailable tool.",
        version="1",
        revision=1,
        required_tools=frozenset({"future_tool"}),
        resources=(),
        metadata={},
    )

    snapshot = AgentSkillCatalogSnapshot.build(
        (skipped, accepted),
        generation=3,
        registered_tools=frozenset({"search_web", "read_web_page"}),
        max_available_skills=8,
    )

    assert tuple(snapshot.skills) == ("web-research",)
    assert snapshot.diagnostics[0].skill_name == "unknown-tools"
    assert snapshot.diagnostics[0].names == ("future_tool",)
    with pytest.raises(TypeError):
        snapshot.skills["other"] = accepted  # type: ignore[index]


def test_skill_catalog_rejects_capacity_without_publishing_a_partial_subset() -> None:
    first = skill()
    second = replace(
        first,
        id=uuid4(),
        name="music-curator",
        display_name="音乐策划",
    )
    with pytest.raises(AgentSkillCatalogCapacityError):
        AgentSkillCatalogSnapshot.build(
            (first, second),
            generation=1,
            registered_tools=frozenset({"search_web", "read_web_page"}),
            max_available_skills=1,
        )


@pytest.mark.asyncio
async def test_reloadable_skill_catalog_uses_ttl_and_generation_before_full_reload() -> None:
    first = skill()
    repository = FakeSkillRepository((first,))
    now = [100.0]
    catalog = ReloadableAgentSkillCatalog(
        repository,
        registered_tools=("search_web", "read_web_page"),
        refresh_seconds=10,
        max_available_skills=8,
        clock=lambda: now[0],
    )

    await catalog.start()
    original = catalog.snapshot
    assert original.loaded is True
    assert repository.generation_calls == 1
    assert repository.load_calls == 1

    assert await catalog.refresh_if_stale() is False
    assert repository.generation_calls == 1

    now[0] = 111
    assert await catalog.refresh_if_stale() is False
    assert repository.generation_calls == 2
    assert repository.load_calls == 1

    replacement = AgentSkill(
        id=first.id,
        name=first.name,
        display_name=first.display_name,
        description="Updated description.",
        instructions=first.instructions,
        version="2",
        revision=2,
        required_tools=first.required_tools,
        resources=first.resources,
        metadata=first.metadata,
    )
    repository.skills = (replacement,)
    repository.generation_value = 2
    now[0] = 122
    assert await catalog.refresh_if_stale() is True
    assert catalog.snapshot is not original
    assert catalog.snapshot.skills["web-research"].version == "2"
    assert repository.load_calls == 2


@pytest.mark.asyncio
async def test_reloadable_skill_catalog_retains_last_snapshot_on_refresh_failure() -> None:
    health = HealthRegistry()
    repository = FakeSkillRepository((skill(),))
    now = [0.0]
    catalog = ReloadableAgentSkillCatalog(
        repository,
        registered_tools=("search_web", "read_web_page"),
        refresh_seconds=5,
        max_available_skills=8,
        health=health,
        clock=lambda: now[0],
    )
    await catalog.start()
    previous = catalog.snapshot

    repository.error = RuntimeError("database unavailable")
    now[0] = 6
    assert await catalog.refresh_if_stale() is False

    assert catalog.snapshot is previous
    skill_health = {item.name: item for item in health.snapshot()}["skills"]
    assert skill_health.state is HealthState.DEGRADED
    assert skill_health.detail == "catalog refresh failed"


def test_skill_availability_requires_loader_and_every_declared_tool() -> None:
    snapshot = AgentSkillCatalogSnapshot.build(
        (skill(),),
        generation=1,
        registered_tools=frozenset({"search_web", "read_web_page"}),
        max_available_skills=8,
    )
    availability = SkillAvailabilityService()

    assert availability.resolve(snapshot, ("search_web", "read_web_page")) == ()
    assert availability.resolve(snapshot, ("load_agent_skill", "search_web")) == ()
    assert [
        item.name
        for item in availability.resolve(
            snapshot,
            ("load_agent_skill", "search_web", "read_web_page"),
        )
    ] == ["web-research"]


def test_known_tool_dependency_is_valid_but_does_not_grant_runtime_availability() -> None:
    music = replace(
        skill(),
        name="music-curator",
        display_name="音乐策划",
        required_tools=frozenset({"list_music_playlists"}),
    )
    snapshot = AgentSkillCatalogSnapshot.build(
        (music,),
        generation=1,
        registered_tools=frozenset({"list_music_playlists"}),
        max_available_skills=8,
    )

    assert snapshot.diagnostics == ()
    assert (
        SkillAvailabilityService().resolve(
            snapshot,
            ("load_agent_skill",),
        )
        == ()
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("key", "../guide", "resource key"),
        ("display_name", "", "display name"),
        ("description", "", "description"),
        ("media_type", "application/octet-stream", "media type"),
        ("content", "", "content"),
        ("position", 0, "position"),
    ],
)
def test_skill_resource_rejects_invalid_text_resource(
    field: str,
    value: object,
    message: str,
) -> None:
    values = {
        "id": uuid4(),
        "key": "guide",
        "display_name": "Guide",
        "description": "Read for detail.",
        "kind": SkillResourceKind.REFERENCE,
        "media_type": "text/plain",
        "content": "Details",
        "position": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        AgentSkillResource(**values)  # type: ignore[arg-type]
