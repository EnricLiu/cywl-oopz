from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from cywl_oopz.commands.builtin import HelpCommand
from cywl_oopz.commands.router import AccessRequirement, CommandRouter, ParsedCommand
from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.features.access.models import (
    AccessResource,
    AccessRole,
    Permission,
    RoleBinding,
    RoleBindingScope,
)
from cywl_oopz.features.access.service import AuthorizationService


@dataclass
class FakeMessage:
    plain_text: str
    text: str = ""
    content: str = ""


class EchoCommand:
    name = "echo"
    description = "Echo input."

    def __init__(self) -> None:
        self.received: ParsedCommand | None = None

    async def execute(self, command: ParsedCommand, _context: object) -> None:
        self.received = command


class FakeRoleBindings:
    def __init__(self, records: tuple[RoleBinding, ...] = ()) -> None:
        self.records = records

    async def list_for_subject(self, subject_person_id: str) -> tuple[RoleBinding, ...]:
        return tuple(
            record for record in self.records if record.subject_person_id == subject_person_id
        )


class UnavailableRoleBindings:
    async def list_for_subject(self, subject_person_id: str):
        del subject_person_id
        raise DatabaseError("unavailable")


class GlobalRebootAccess:
    def is_available(self, invocation):
        del invocation
        return True

    def requirement(self, command, invocation):
        del command, invocation
        return AccessRequirement(Permission.BOT_REBOOT, AccessResource.global_resource())

    def visibility_requirement(self, invocation):
        del invocation
        return AccessRequirement(Permission.BOT_REBOOT, AccessResource.global_resource())


class FakeContext:
    def __init__(self, person_id: str = "person") -> None:
        message = SimpleNamespace(
            sender_id=person_id,
            area="area",
            channel="channel",
        )
        self.event = SimpleNamespace(message=message, is_private=False)
        self.replies: list[str] = []

    async def reply(self, text: str) -> None:
        self.replies.append(text)


def test_parse_requires_prefix() -> None:
    router = CommandRouter("/")

    assert router.parse("hello") is None
    assert router.parse("/") is None
    assert router.parse("/ECHO one two") == ParsedCommand("echo", ("one", "two"))


def test_parse_preserves_raw_json_argument_spacing() -> None:
    router = CommandRouter("/")

    parsed = router.parse('/tool echo_debug {"value": "one  two"}')

    assert parsed is not None
    assert parsed.name == "tool"
    assert parsed.raw_arguments == 'echo_debug {"value": "one  two"}'


@pytest.mark.asyncio
async def test_dispatch_executes_registered_command() -> None:
    router = CommandRouter("/")
    command = EchoCommand()
    router.register(command)

    consumed = await router.dispatch(FakeMessage("/echo hello"), object())

    assert consumed is True
    assert command.received == ParsedCommand("echo", ("hello",))


@pytest.mark.asyncio
async def test_dispatch_prefers_raw_text_before_plain_text_removes_mentions() -> None:
    router = CommandRouter("/")
    command = EchoCommand()
    router.register(command)
    message = FakeMessage(
        "/echo leftright",
        text="/echo left(met)target(met)right",
    )

    consumed = await router.dispatch(message, object())

    assert consumed is True
    assert command.received == ParsedCommand(
        "echo",
        ("left(met)target(met)right",),
    )


@pytest.mark.asyncio
async def test_restricted_dispatch_denies_without_matching_global_role() -> None:
    authorizer = AuthorizationService(
        FakeRoleBindings(
            (
                RoleBinding(
                    "person",
                    AccessRole.ADMIN,
                    RoleBindingScope.AREA,
                    area_id="area",
                ),
            )
        )
    )
    router = CommandRouter("/", authorizer)
    command = EchoCommand()
    command.name = "reboot"
    router.register(command, access=GlobalRebootAccess())
    context = FakeContext()

    consumed = await router.dispatch(FakeMessage("/reboot"), context)

    assert consumed is True
    assert command.received is None
    assert context.replies == ["你没有执行此操作的权限。"]


@pytest.mark.asyncio
async def test_help_filters_restricted_commands_with_same_policy() -> None:
    authorizer = AuthorizationService(FakeRoleBindings())
    router = CommandRouter("/", authorizer)
    router.register(EchoCommand())
    reboot = EchoCommand()
    reboot.name = "reboot"
    reboot.description = "Restart."
    router.register(reboot, access=GlobalRebootAccess())
    router.register(HelpCommand(router))
    context = FakeContext()

    await router.dispatch(FakeMessage("/help"), context)

    assert "/echo" in context.replies[0]
    assert "/help" in context.replies[0]
    assert "/reboot" not in context.replies[0]


@pytest.mark.asyncio
async def test_help_fresh_reads_role_changes_on_every_invocation() -> None:
    bindings = FakeRoleBindings()
    authorizer = AuthorizationService(bindings)
    router = CommandRouter("/", authorizer)
    reboot = EchoCommand()
    reboot.name = "reboot"
    reboot.description = "Restart."
    router.register(reboot, access=GlobalRebootAccess())
    router.register(HelpCommand(router))

    before = FakeContext()
    await router.dispatch(FakeMessage("/help"), before)
    assert "/reboot" not in before.replies[0]

    bindings.records = (
        RoleBinding(
            "person",
            AccessRole.ADMIN,
            RoleBindingScope.GLOBAL,
        ),
    )
    after_grant = FakeContext()
    await router.dispatch(FakeMessage("/help"), after_grant)
    assert "/reboot" in after_grant.replies[0]

    bindings.records = ()
    after_revoke = FakeContext()
    await router.dispatch(FakeMessage("/help"), after_revoke)
    assert "/reboot" not in after_revoke.replies[0]


@pytest.mark.asyncio
async def test_restricted_dispatch_fails_closed_when_role_store_is_unavailable() -> None:
    router = CommandRouter("/", AuthorizationService(UnavailableRoleBindings()))
    command = EchoCommand()
    command.name = "reboot"
    router.register(command, access=GlobalRebootAccess())
    context = FakeContext()

    consumed = await router.dispatch(FakeMessage("/reboot"), context)

    assert consumed is True
    assert command.received is None
    assert context.replies == ["权限服务暂时不可用，请稍后重试。"]
