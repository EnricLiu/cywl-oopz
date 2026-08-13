from __future__ import annotations

from dataclasses import dataclass

import pytest

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    CommandDefinition,
    CommandUsageError,
    NoArgumentsParser,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.models import (
    CommandActor,
    CommandLocation,
    CommandRequest,
    CommandScope,
    CommandSource,
    CommandText,
    CommandTrigger,
    DispatchStatus,
)
from cywl_oopz.commands.router import CommandRouter


class FakeResponder:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def reply(self, text: str) -> None:
        self.replies.append(text)

    async def send(self, text: str) -> None:
        self.replies.append(text)

    async def react(self, emoji: str) -> None:
        del emoji


def request(tokens: tuple[str, ...]) -> tuple[CommandRequest, FakeResponder]:
    responder = FakeResponder()
    raw_tail = " ".join(tokens)
    return (
        CommandRequest(
            trigger=CommandTrigger.TEXT,
            actor=CommandActor("actor"),
            location=CommandLocation(CommandScope.CHANNEL, "area", "channel"),
            source=CommandSource("message"),
            responder=responder,
            text=CommandText(
                f"/typed {raw_tail}".rstrip(),
                "typed",
                raw_tail,
                tokens,
            ),
        ),
        responder,
    )


@dataclass(frozen=True, slots=True)
class TypedArguments:
    value: str


class RecordingParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, command_request: CommandRequest) -> TypedArguments:
        self.calls += 1
        assert command_request.text is not None
        if len(command_request.text.tokens) != 1:
            raise CommandUsageError("需要一个参数。")
        return TypedArguments(command_request.text.tokens[0])


class RecordingAuthorization:
    def __init__(self) -> None:
        self.arguments: list[TypedArguments] = []

    def is_available(self, command_request: CommandRequest) -> bool:
        del command_request
        return True

    def requirement(self, command_request, arguments):
        del command_request
        self.arguments.append(arguments)
        return None

    def visibility_requirement(self, command_request):
        del command_request
        return None


class RecordingHandler:
    name = "typed"
    description = "Typed."

    def __init__(self) -> None:
        self.arguments: list[TypedArguments] = []

    async def handle(self, command_request, arguments):
        del command_request
        self.arguments.append(arguments)


class NoArgumentsHandler:
    def __init__(self) -> None:
        self.calls = 0

    async def handle(self, command_request, arguments):
        del command_request, arguments
        self.calls += 1


@pytest.mark.asyncio
async def test_typed_definition_parses_once_and_shares_the_same_arguments() -> None:
    parser = RecordingParser()
    authorization = RecordingAuthorization()
    handler = RecordingHandler()
    router = CommandRouter("/")
    router.register_definition(
        CommandDefinition(
            CommandSpec("typed", "Typed.", "test", ("typed <value>",)),
            parser,
            handler,
            authorization,
        )
    )
    command_request, _ = request(("value",))

    outcome = await router.dispatch_request(command_request, object())

    assert outcome.status is DispatchStatus.COMPLETED
    assert parser.calls == 1
    assert authorization.arguments == [TypedArguments("value")]
    assert handler.arguments == [TypedArguments("value")]
    assert authorization.arguments[0] is handler.arguments[0]


@pytest.mark.asyncio
async def test_usage_error_stops_before_authorization_and_handler() -> None:
    parser = RecordingParser()
    authorization = RecordingAuthorization()
    handler = RecordingHandler()
    router = CommandRouter("!")
    router.register_definition(
        CommandDefinition(
            CommandSpec("typed", "Typed.", "test", ("typed <value>",)),
            parser,
            handler,
            authorization,
        )
    )
    command_request, responder = request(())

    outcome = await router.dispatch_request(command_request, object())

    assert outcome.status is DispatchStatus.COMPLETED
    assert authorization.arguments == []
    assert handler.arguments == []
    assert responder.replies == ["需要一个参数。\n用法：\n!typed <value>"]


@pytest.mark.asyncio
async def test_no_arguments_parser_rejects_before_state_change() -> None:
    handler = NoArgumentsHandler()
    router = CommandRouter("/")
    router.register_definition(
        CommandDefinition(
            CommandSpec("typed", "Typed.", "test", ("typed",)),
            NoArgumentsParser(),
            handler,
            PublicCommandAuthorization(),
        )
    )
    command_request, responder = request(("unexpected",))

    await router.dispatch_request(command_request, object())

    assert handler.calls == 0
    assert responder.replies == ["此命令不接受额外参数。\n用法：\n/typed"]
