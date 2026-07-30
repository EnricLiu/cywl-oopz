from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cywl_oopz.features.agent.skills.errors import AgentSkillLibraryError
from cywl_oopz.features.agent.skills.library import AgentSkillLibraryService
from cywl_oopz.features.agent.skills.models import (
    AgentSkill,
    AgentSkillDiscovery,
    AgentSkillResource,
    AgentSkillRevokeResult,
    AgentSkillShare,
    AgentSkillShareSummary,
    SkillAccessKind,
    SkillOwnershipKind,
    SkillResourceKind,
    SkillShareStatus,
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
        "invite_many": AsyncMock(return_value=()),
        "pending_invitations": AsyncMock(return_value=()),
        "outgoing_shares": AsyncMock(return_value=()),
        "share_for_recipient": AsyncMock(return_value=None),
        "share_for_owner": AsyncMock(return_value=None),
        "respond": AsyncMock(),
        "revoke": AsyncMock(return_value=None),
        "revoke_owned_shares": AsyncMock(return_value=()),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def service(
    repository,
    *,
    max_personal: int = 2,
    max_resources: int = 2,
    max_shared: int = 8,
    max_recipients: int = 5,
    notifier=None,
):
    return AgentSkillLibraryService(
        repository,
        registered_tools=frozenset({"search_web", "load_agent_skill"}),
        max_personal_skills=max_personal,
        max_available_skills=8,
        max_resources_per_skill=max_resources,
        max_instruction_characters=1000,
        max_resource_characters=1000,
        max_accepted_shared_skills=max_shared,
        max_share_recipients_per_call=max_recipients,
        notifier=notifier,
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


def share_summary(
    skill: AgentSkill,
    *,
    recipient: str = "friend",
    status: SkillShareStatus = SkillShareStatus.PENDING,
    active: bool = True,
) -> AgentSkillShareSummary:
    now = datetime.now(UTC)
    share = AgentSkillShare(
        id=uuid4(),
        skill_id=skill.id,
        recipient_person_id=recipient,
        status=status,
        created_at=now,
        updated_at=now,
        responded_at=None if status is SkillShareStatus.PENDING else now,
    )
    discovery = AgentSkillDiscovery(
        id=skill.id,
        name=skill.name,
        display_name=skill.display_name,
        description=skill.description,
        version=skill.version,
        revision=skill.revision,
        required_tools=skill.required_tools,
        access=SkillAccessKind.SHARED,
    )
    return AgentSkillShareSummary(share, discovery, active)


@pytest.mark.asyncio
async def test_library_service_invites_only_trusted_mentions_and_keeps_notification_failures() -> (
    None
):
    skill = personal_skill()
    first = share_summary(skill, recipient="friend-one").share
    second = share_summary(skill, recipient="friend-two").share
    repository = store(
        get_owned=AsyncMock(return_value=skill),
        invite_many=AsyncMock(return_value=(first, second)),
    )
    notifier = SimpleNamespace(
        invitation=AsyncMock(side_effect=(True, False)),
        revoked=AsyncMock(return_value=True),
    )
    library = service(repository, notifier=notifier)

    result = await library.invite(
        "person",
        skill.id,
        ("friend-one", "friend-one", "person", "friend-two"),
    )

    assert len(result.shares) == 2
    assert result.notification_failures == 1
    assert repository.invite_many.await_args.args[2] == ("friend-one", "friend-two")
    assert notifier.invitation.await_count == 2


@pytest.mark.asyncio
async def test_library_service_rejects_missing_or_excessive_share_mentions() -> None:
    skill = personal_skill()
    repository = store(get_owned=AsyncMock(return_value=skill))
    library = service(repository, max_recipients=1)

    with pytest.raises(AgentSkillLibraryError, match="skill_share_target_required"):
        await library.invite("person", skill.id, ("person",))
    with pytest.raises(AgentSkillLibraryError, match="skill_share_target_limit"):
        await library.invite("person", skill.id, ("one", "two"))
    repository.invite_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_library_service_accepts_pending_invitation_and_rejects_status_flip() -> None:
    skill = personal_skill()
    pending = share_summary(skill)
    accepted = share_summary(skill, status=SkillShareStatus.ACCEPTED)
    repository = store(
        share_for_recipient=AsyncMock(side_effect=(pending, accepted)),
    )
    library = service(repository)

    result = await library.respond("friend", pending.share.id, accepted=True)

    assert result.share.status is SkillShareStatus.ACCEPTED
    repository.respond.assert_awaited_once()

    repository.share_for_recipient.side_effect = None
    repository.share_for_recipient.return_value = accepted
    with pytest.raises(AgentSkillLibraryError, match="skill_invitation_answered"):
        await library.respond("friend", accepted.share.id, accepted=False)


@pytest.mark.asyncio
async def test_library_service_enforces_shared_capacity_and_revokes_best_effort() -> None:
    skill = personal_skill()
    pending = share_summary(skill)
    repository = store(
        share_for_recipient=AsyncMock(return_value=pending),
        list_accessible=AsyncMock(
            return_value=tuple(replace(pending.skill, id=uuid4()) for _ in range(2))
        ),
    )
    limited = service(repository, max_shared=2)
    with pytest.raises(AgentSkillLibraryError, match="skill_shared_library_limit"):
        await limited.respond("friend", pending.share.id, accepted=True)

    repository = store(
        get_owned=AsyncMock(return_value=skill),
        revoke_owned_shares=AsyncMock(return_value=(pending.share,)),
    )
    notifier = SimpleNamespace(
        invitation=AsyncMock(return_value=True),
        revoked=AsyncMock(return_value=False),
    )
    library = service(repository, notifier=notifier)

    result = await library.revoke(
        "person",
        skill.id,
        ("friend",),
        revoke_all=False,
    )

    assert isinstance(result, AgentSkillRevokeResult)
    assert result.notification_failures == 1
    notifier.revoked.assert_awaited_once()


@pytest.mark.asyncio
async def test_library_service_requires_mentions_unless_all_shares_are_explicitly_revoked() -> None:
    skill = personal_skill()
    share = share_summary(skill).share
    repository = store(
        get_owned=AsyncMock(return_value=skill),
        revoke_owned_shares=AsyncMock(return_value=(share,)),
    )
    library = service(repository)

    with pytest.raises(AgentSkillLibraryError, match="skill_share_target_required"):
        await library.revoke("person", skill.id, (), revoke_all=False)
    with pytest.raises(AgentSkillLibraryError, match="skill_share_target_conflict"):
        await library.revoke("person", skill.id, ("friend",), revoke_all=True)

    result = await library.revoke("person", skill.id, (), revoke_all=True)

    assert len(result.shares) == 1
    repository.revoke_owned_shares.assert_awaited_once_with(
        "person",
        skill.id,
        None,
    )
