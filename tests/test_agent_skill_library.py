from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cywl_oopz.features.agent.skills.errors import AgentSkillLibraryError
from cywl_oopz.features.agent.skills.library import AgentSkillLibraryService
from cywl_oopz.features.agent.skills.models import (
    AgentSkill,
    AgentSkillResource,
    SkillOwnershipKind,
    SkillResourceKind,
)


def personal_skill(
    *,
    revision: int = 1,
    resources: tuple[AgentSkillResource, ...] = (),
) -> AgentSkill:
    return AgentSkill(
        id=uuid4(),
        name="travel-planner",
        display_name="旅行规划",
        description="规划行程时使用。",
        instructions="先确认目的地和日期，再整理每日安排。",
        version="1",
        revision=revision,
        required_tools=frozenset({"search_web"}),
        resources=resources,
        metadata={},
        ownership_kind=SkillOwnershipKind.PERSONAL,
        owner_person_id="person",
    )


def resource() -> AgentSkillResource:
    return AgentSkillResource(
        id=uuid4(),
        key="packing-list",
        display_name="行李清单",
        description="整理出发物品时读取。",
        kind=SkillResourceKind.TEMPLATE,
        media_type="text/markdown",
        content="# 清单\n- 证件",
        position=1,
    )


def store(**overrides):
    defaults = {
        "list_owned": AsyncMock(return_value=()),
        "list_accessible": AsyncMock(return_value=()),
        "inspect_accessible": AsyncMock(return_value=None),
        "read_inspectable_resource": AsyncMock(return_value=None),
        "add_personal": AsyncMock(),
        "get_owned": AsyncMock(return_value=None),
        "update_owned": AsyncMock(),
        "upsert_owned_resource": AsyncMock(),
        "remove_owned_resource": AsyncMock(),
        "set_owned_state": AsyncMock(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def service(repository, *, max_personal: int = 2, max_resources: int = 2):
    return AgentSkillLibraryService(
        repository,
        registered_tools=frozenset({"search_web", "load_agent_skill"}),
        max_personal_skills=max_personal,
        max_available_skills=8,
        max_resources_per_skill=max_resources,
        max_instruction_characters=1000,
        max_resource_characters=1000,
    )


@pytest.mark.asyncio
async def test_library_service_creates_private_skill_from_trusted_owner() -> None:
    repository = store()
    library = service(repository)

    created = await library.create(
        " person ",
        name="travel-planner",
        display_name="旅行规划",
        description="规划行程时使用。",
        instructions="先确认目的地和日期，再整理每日安排。",
        required_tools=frozenset({"search_web"}),
    )

    assert created.owner_person_id == "person"
    assert created.ownership_kind is SkillOwnershipKind.PERSONAL
    assert created.resources == ()
    repository.add_personal.assert_awaited_once_with(created)


@pytest.mark.asyncio
async def test_library_service_enforces_capacity_tools_owner_and_revision() -> None:
    current = personal_skill(revision=2)
    full_repository = store(
        list_owned=AsyncMock(return_value=(object(), object())),
        get_owned=AsyncMock(return_value=current),
    )
    library = service(full_repository, max_personal=2)

    with pytest.raises(AgentSkillLibraryError, match="skill_library_limit"):
        await library.create(
            "person",
            name="other",
            display_name="Other",
            description="Other workflow.",
            instructions="Do the workflow.",
            required_tools=frozenset(),
        )
    full_repository.list_owned.return_value = ()
    with pytest.raises(AgentSkillLibraryError, match="skill_unknown_required_tools"):
        await library.create(
            "person",
            name="other",
            display_name="Other",
            description="Other workflow.",
            instructions="Do the workflow.",
            required_tools=frozenset({"invented_tool"}),
        )
    with pytest.raises(AgentSkillLibraryError, match="skill_revision_conflict"):
        await library.update("person", current.id, 1, version="2")

    full_repository.get_owned.return_value = None
    with pytest.raises(AgentSkillLibraryError, match="skill_not_owned"):
        await library.update("person", current.id, 2, version="2")


@pytest.mark.asyncio
async def test_library_service_updates_and_bounds_resources_without_noop_writes() -> None:
    current = personal_skill(revision=2)
    updated = replace(current, version="2", revision=3)
    repository = store(
        get_owned=AsyncMock(return_value=current),
        update_owned=AsyncMock(return_value=updated),
    )
    library = service(repository, max_resources=1)

    result = await library.update("person", current.id, 2, version="2")

    assert result is updated
    proposed = repository.update_owned.await_args.args[0]
    assert proposed.version == "2"
    assert proposed.revision == 2
    with pytest.raises(AgentSkillLibraryError, match="skill_no_changes"):
        await library.update("person", current.id, 2, version="1")

    existing = resource()
    repository.get_owned.return_value = replace(current, resources=(existing,))
    with pytest.raises(AgentSkillLibraryError, match="skill_resource_library_limit"):
        await library.upsert_resource(
            "person",
            current.id,
            2,
            key="second",
            display_name="Second",
            description="Second resource.",
            kind=SkillResourceKind.REFERENCE,
            media_type="text/plain",
            content="Second content.",
            position=2,
        )
    with pytest.raises(AgentSkillLibraryError, match="skill_no_changes"):
        await library.upsert_resource(
            "person",
            current.id,
            2,
            key=existing.key,
            display_name=existing.display_name,
            description=existing.description,
            kind=existing.kind,
            media_type=existing.media_type,
            content=existing.content,
            position=existing.position,
        )
