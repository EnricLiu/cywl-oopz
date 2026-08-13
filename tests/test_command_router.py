from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from cywl_oopz.commands.builtin import HelpCommand, PingCommand
from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    AccessRequirement,
    CommandDefinition,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.models import CommandRequest, CommandText
from cywl_oopz.commands.parsing import CommandTextParser
from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.features.access.models import (
    AccessResource,
    AccessRole,
    Permission,
    RoleBinding,
    RoleBindingScope,
)
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.testing.commands import dispatch_command


@dataclass
class FakeMessage:
    plain_text: str
    text: str = ""
    content: str = ""
    sender_id: str = "person"
    area: str = "area"
    channel: str = "channel"


class EchoCommand:
    name = "echo"
    description = "Echo input."

    def __init__(self) -> None:
        self.received: CommandText | None = None

    def definition(self, authorization=None) -> CommandDefinition[CommandText]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            self,
            self,
            authorization or PublicCommandAuthorization(),
        )

    def parse(self, request: CommandRequest) -> CommandText:
        assert request.text is not None
        return request.text

    async def handle(self, request: CommandRequest, command: CommandText) -> None:
        del request
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


class GlobalRebootAuthorization:
    def is_available(self, request):
        del request
        return True

    def requirement(self, request, arguments):
        del request, arguments
        return AccessRequirement(Permission.BOT_REBOOT, AccessResource.global_resource())

    def visibility_requirement(self, request):
        del request
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
    parser = CommandTextParser("/")

    assert parser.parse("hello") is None
    assert parser.parse("/") is None
    parsed = parser.parse("/ECHO one two")
    assert parsed is not None
    assert parsed.name == "echo"
    assert parsed.tokens == ("one", "two")


def test_parse_preserves_raw_json_argument_spacing() -> None:
    parser = CommandTextParser("/")

    parsed = parser.parse('/tool echo_debug {"value": "one  two"}')

    assert parsed is not None
    assert parsed.name == "tool"
    assert parsed.raw_tail == 'echo_debug {"value": "one  two"}'


@pytest.mark.asyncio
async def test_dispatch_executes_registered_command() -> None:
    router = CommandRouter("/")
    command = EchoCommand()
    router.register_definition(command.definition())

    consumed = await dispatch_command(router, FakeMessage("/echo hello"), object())

    assert consumed is True
    assert command.received is not None
    assert command.received.tokens == ("hello",)


@pytest.mark.asyncio
async def test_typed_ping_rejects_extra_arguments() -> None:
    router = CommandRouter("/")
    router.register_definition(PingCommand().definition())
    context = FakeContext()

    await dispatch_command(router, FakeMessage("/ping extra"), context)

    assert context.replies == ["此命令不接受额外参数。\n用法：/ping"]


@pytest.mark.asyncio
async def test_dispatch_prefers_raw_text_before_plain_text_removes_mentions() -> None:
    router = CommandRouter("/")
    command = EchoCommand()
    router.register_definition(command.definition())
    message = FakeMessage(
        "/echo leftright",
        text="/echo left(met)target(met)right",
    )

    consumed = await dispatch_command(router, message, object())

    assert consumed is True
    assert command.received is not None
    assert command.received.tokens == ("left(met)target(met)right",)


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
    router.register_definition(command.definition(GlobalRebootAuthorization()))
    context = FakeContext()

    consumed = await dispatch_command(router, FakeMessage("/reboot"), context)

    assert consumed is True
    assert command.received is None
    assert context.replies == ["你没有执行此操作的权限。"]


@pytest.mark.asyncio
async def test_help_filters_restricted_commands_with_same_policy() -> None:
    authorizer = AuthorizationService(FakeRoleBindings())
    router = CommandRouter("/", authorizer)
    router.register_definition(EchoCommand().definition())
    reboot = EchoCommand()
    reboot.name = "reboot"
    reboot.description = "Restart."
    router.register_definition(reboot.definition(GlobalRebootAuthorization()))
    router.register_definition(HelpCommand(router).definition())
    context = FakeContext()

    await dispatch_command(router, FakeMessage("/help"), context)

    assert "/echo" in context.replies[0]
    assert "/help" in context.replies[0]
    assert "/reboot" not in context.replies[0]


@pytest.mark.asyncio
async def test_help_renders_dynamic_prefix_and_detailed_command_metadata() -> None:
    router = CommandRouter("!")
    echo = EchoCommand()
    echo.category = "测试"
    echo.usage = ("echo <内容>",)
    echo.examples = ("echo 你好",)
    router.register_definition(echo.definition())
    router.register_definition(HelpCommand(router).definition())
    context = FakeContext()

    await dispatch_command(router, FakeMessage("!help echo"), context)

    assert "**!echo**" in context.replies[0]
    assert "!echo <内容>" in context.replies[0]
    assert "!echo 你好" in context.replies[0]
    assert "/echo" not in context.replies[0]


@pytest.mark.asyncio
async def test_help_rejects_extra_arguments_with_dynamic_prefix() -> None:
    router = CommandRouter("!")
    router.register_definition(HelpCommand(router).definition())
    context = FakeContext()

    await dispatch_command(router, FakeMessage("!help one two"), context)

    assert context.replies == ["用法：\n!help\n!help <命令>"]


@pytest.mark.asyncio
async def test_help_fresh_reads_role_changes_on_every_invocation() -> None:
    bindings = FakeRoleBindings()
    authorizer = AuthorizationService(bindings)
    router = CommandRouter("/", authorizer)
    reboot = EchoCommand()
    reboot.name = "reboot"
    reboot.description = "Restart."
    router.register_definition(reboot.definition(GlobalRebootAuthorization()))
    router.register_definition(HelpCommand(router).definition())

    before = FakeContext()
    await dispatch_command(router, FakeMessage("/help"), before)
    assert "/reboot" not in before.replies[0]

    bindings.records = (
        RoleBinding(
            "person",
            AccessRole.ADMIN,
            RoleBindingScope.GLOBAL,
        ),
    )
    after_grant = FakeContext()
    await dispatch_command(router, FakeMessage("/help"), after_grant)
    assert "/reboot" in after_grant.replies[0]

    bindings.records = ()
    after_revoke = FakeContext()
    await dispatch_command(router, FakeMessage("/help"), after_revoke)
    assert "/reboot" not in after_revoke.replies[0]


@pytest.mark.asyncio
async def test_restricted_dispatch_fails_closed_when_role_store_is_unavailable() -> None:
    router = CommandRouter("/", AuthorizationService(UnavailableRoleBindings()))
    command = EchoCommand()
    command.name = "reboot"
    router.register_definition(command.definition(GlobalRebootAuthorization()))
    context = FakeContext()

    consumed = await dispatch_command(router, FakeMessage("/reboot"), context)

    assert consumed is True
    assert command.received is None
    assert context.replies == ["权限服务暂时不可用，请稍后重试。"]
