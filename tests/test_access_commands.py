from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from cywl_oopz.commands.parsing import CommandTextParser
from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.features.access.administration import RoleAdministrationService
from cywl_oopz.features.access.commands import (
    RoleArgumentsParser,
    RoleCommand,
    RoleCommandAuthorization,
    WhoAmICommand,
)
from cywl_oopz.features.access.models import (
    AccessResourceKind,
    AccessRole,
    RoleBinding,
    RoleBindingScope,
)
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.integrations.oopz.command_requests import OopzCommandRequestFactory


@dataclass
class InMemoryRoleRepository:
    records: list[RoleBinding] = field(default_factory=list)

    async def list_for_subject(self, subject_person_id: str) -> tuple[RoleBinding, ...]:
        return tuple(
            record for record in self.records if record.subject_person_id == subject_person_id
        )

    async def list_bindings(
        self,
        *,
        subject_person_id: str | None = None,
    ) -> tuple[RoleBinding, ...]:
        return tuple(
            record
            for record in self.records
            if subject_person_id is None or record.subject_person_id == subject_person_id
        )

    async def grant(self, binding: RoleBinding) -> bool:
        key = (
            binding.subject_person_id,
            binding.role,
            binding.scope,
            binding.area_id,
            binding.channel_id,
        )
        if any(
            (
                current.subject_person_id,
                current.role,
                current.scope,
                current.area_id,
                current.channel_id,
            )
            == key
            for current in self.records
        ):
            return False
        self.records.append(binding)
        return True

    async def revoke(
        self,
        subject_person_id: str,
        role: AccessRole,
        scope: RoleBindingScope,
        *,
        area_id: str = "",
        channel_id: str = "",
    ) -> bool:
        for index, current in enumerate(self.records):
            if (
                current.subject_person_id == subject_person_id
                and current.role is role
                and current.scope is scope
                and current.area_id == area_id
                and current.channel_id == channel_id
            ):
                self.records.pop(index)
                return True
        return False


class FakeMessage:
    def __init__(
        self,
        text: str,
        person_id: str,
        *,
        mentions: tuple[str, ...] = (),
        plain_text: str | None = None,
    ) -> None:
        self.plain_text = text if plain_text is None else plain_text
        self.text = text
        self.content = text
        self.sender_id = person_id
        self.area = "area"
        self.channel = "channel"
        self.mention_list = [SimpleNamespace(person=person_id) for person_id in mentions]


class FakeContext:
    def __init__(self, message: FakeMessage, *, private: bool = False) -> None:
        self.event = SimpleNamespace(message=message, is_private=private)
        self.config = SimpleNamespace(person_uid="bot")
        self.replies: list[str] = []

    async def reply(self, text: str) -> None:
        self.replies.append(text)


def router_and_repository() -> tuple[CommandRouter, InMemoryRoleRepository]:
    repository = InMemoryRoleRepository()
    authorizer = AuthorizationService(repository, frozenset({"owner"}))
    administration = RoleAdministrationService(repository, authorizer)
    router = CommandRouter("/", authorizer)
    router.register_definition(WhoAmICommand().definition())
    router.register_definition(RoleCommand(authorizer, administration).definition())
    return router, repository


@pytest.mark.asyncio
async def test_whoami_reports_exact_message_sender_id() -> None:
    router, _ = router_and_repository()
    message = FakeMessage("/whoami", "person-42")
    context = FakeContext(message)

    assert await router.dispatch(message, context)

    assert context.replies == ["你的 OOPZ ID：person-42"]


@pytest.mark.asyncio
async def test_bootstrap_owner_grants_area_admin_and_change_is_immediate() -> None:
    router, repository = router_and_repository()
    grant = FakeMessage("/role grant admin area", "owner", mentions=("target",))
    grant_context = FakeContext(grant)

    assert await router.dispatch(grant, grant_context)

    assert grant_context.replies == ["已授予：admin · area"]
    assert repository.records == [
        RoleBinding(
            subject_person_id="target",
            role=AccessRole.ADMIN,
            scope=RoleBindingScope.AREA,
            area_id="area",
            granted_by_person_id="owner",
        )
    ]

    me = FakeMessage("/role me", "target")
    me_context = FakeContext(me)
    assert await router.dispatch(me, me_context)
    assert "当前角色：admin" in me_context.replies[0]
    assert "channel.initialize" in me_context.replies[0]
    assert "rbac.manage" not in me_context.replies[0]


@pytest.mark.asyncio
async def test_role_grant_uses_mention_list_and_ignores_oopz_inline_marker() -> None:
    router, repository = router_and_repository()
    target = "96610cfa481f11ef887a2ab75f6d1b3b"
    message = FakeMessage(
        f"/role grant(met){target}(met)admin area",
        "owner",
        mentions=(target,),
        plain_text="/role grantadmin area",
    )
    context = FakeContext(message)

    assert await router.dispatch(message, context)

    assert context.replies == ["已授予：admin · area"]
    assert repository.records == [
        RoleBinding(
            subject_person_id=target,
            role=AccessRole.ADMIN,
            scope=RoleBindingScope.AREA,
            area_id="area",
            granted_by_person_id="owner",
        )
    ]


def test_role_grant_access_scope_ignores_oopz_inline_mention_marker() -> None:
    target = "96610cfa481f11ef887a2ab75f6d1b3b"
    message = FakeMessage(
        f"/role grant(met){target}(met)admin area",
        "owner",
        mentions=(target,),
        plain_text="/role grantadmin area",
    )
    request = OopzCommandRequestFactory(CommandTextParser("/")).from_message(
        message,
        FakeContext(message),
    )
    assert request is not None
    arguments = RoleArgumentsParser().parse(request)

    requirement = RoleCommandAuthorization().requirement(request, arguments)

    assert requirement is not None
    assert requirement.resource.kind is AccessResourceKind.AREA
    assert requirement.resource.area_id == "area"


@pytest.mark.asyncio
async def test_area_admin_cannot_grant_roles() -> None:
    router, repository = router_and_repository()
    repository.records.append(
        RoleBinding(
            subject_person_id="area-admin",
            role=AccessRole.ADMIN,
            scope=RoleBindingScope.AREA,
            area_id="area",
        )
    )
    message = FakeMessage(
        "/role grant moderator channel",
        "area-admin",
        mentions=("target",),
    )
    context = FakeContext(message)

    assert await router.dispatch(message, context)

    assert context.replies == ["你没有执行此操作的权限。"]
    assert len(repository.records) == 1


@pytest.mark.asyncio
async def test_role_mutation_requires_one_real_mention() -> None:
    router, repository = router_and_repository()
    message = FakeMessage("/role grant admin area", "owner")
    context = FakeContext(message)

    assert await router.dispatch(message, context)

    assert context.replies == ["请在当前消息中准确 @ 一位目标用户。"]
    assert repository.records == []


@pytest.mark.asyncio
async def test_bootstrap_owner_cannot_be_revoked_from_chat() -> None:
    router, repository = router_and_repository()
    message = FakeMessage(
        "/role revoke owner global",
        "owner",
        mentions=("owner",),
    )
    context = FakeContext(message)

    assert await router.dispatch(message, context)

    assert context.replies == ["Bootstrap owner 不能通过命令撤销，请修改本地环境配置。"]
    assert repository.records == []
