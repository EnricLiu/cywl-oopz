from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

import pytest

from cywl_oopz.features.agent.skills.availability import (
    SkillAvailabilityCapacityError,
    SkillAvailabilityService,
)
from cywl_oopz.features.agent.skills.models import (
    AgentSkill,
    AgentSkillDiscovery,
    AgentSkillResource,
    AgentSkillShare,
    SkillAccessKind,
    SkillOwnershipKind,
    SkillResourceKind,
    SkillShareStatus,
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


def discovery_of(value: AgentSkill) -> AgentSkillDiscovery:
    return AgentSkillDiscovery(
        id=value.id,
        name=value.name,
        display_name=value.display_name,
        description=value.description,
        version=value.version,
        revision=value.revision,
        required_tools=value.required_tools,
        access=SkillAccessKind.BUILTIN,
    )


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


def test_personal_skill_requires_one_owner_and_archive_disables_it() -> None:
    personal = replace(
        skill(),
        ownership_kind=SkillOwnershipKind.PERSONAL,
        owner_person_id=" person ",
    )

    assert personal.owner_person_id == "person"
    assert personal.ownership_kind is SkillOwnershipKind.PERSONAL

    with pytest.raises(ValueError, match="must not have an owner"):
        replace(skill(), owner_person_id="person")
    with pytest.raises(ValueError, match="requires an owner"):
        replace(skill(), ownership_kind=SkillOwnershipKind.PERSONAL)
    with pytest.raises(ValueError, match="must not be enabled"):
        replace(
            personal,
            archived_at=datetime.now(UTC),
            enabled=True,
        )
    with pytest.raises(ValueError, match="must be enabled"):
        replace(personal, enabled=False)


def test_discovery_and_share_values_validate_user_visible_identity() -> None:
    source = skill()
    discovery = AgentSkillDiscovery(
        id=source.id,
        name=source.name,
        display_name=source.display_name,
        description=source.description,
        version=source.version,
        revision=source.revision,
        required_tools=source.required_tools,
        access=SkillAccessKind.BUILTIN,
    )
    now = datetime.now(UTC)
    pending = AgentSkillShare(
        id=uuid4(),
        skill_id=source.id,
        recipient_person_id=" recipient ",
        status=SkillShareStatus.PENDING,
        created_at=now,
        updated_at=now,
    )

    assert discovery.access is SkillAccessKind.BUILTIN
    assert pending.recipient_person_id == "recipient"
    with pytest.raises(ValueError, match="must not have a response"):
        replace(pending, responded_at=now)
    with pytest.raises(ValueError, match="requires a response"):
        replace(pending, status=SkillShareStatus.ACCEPTED)


def test_skill_availability_requires_loader_and_every_declared_tool() -> None:
    discoveries = (discovery_of(skill()),)
    availability = SkillAvailabilityService()

    assert availability.resolve(discoveries, ("search_web", "read_web_page")) == ()
    assert availability.resolve(discoveries, ("load_agent_skill", "search_web")) == ()
    assert [
        item.name
        for item in availability.resolve(
            discoveries,
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
    discoveries = (discovery_of(music),)
    assert (
        SkillAvailabilityService().resolve(
            discoveries,
            ("load_agent_skill",),
        )
        == ()
    )


def test_skill_availability_rejects_an_oversized_user_library() -> None:
    first = discovery_of(skill())
    second = replace(
        first,
        id=uuid4(),
        name="music-curator",
        display_name="音乐策划",
    )

    with pytest.raises(SkillAvailabilityCapacityError):
        SkillAvailabilityService(max_available_skills=1).resolve(
            (first, second),
            ("load_agent_skill", "search_web", "read_web_page"),
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
