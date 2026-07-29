from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from uuid import uuid4

import pytest

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
