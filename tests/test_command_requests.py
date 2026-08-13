from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from cywl_oopz.commands.models import CommandScope, DispatchStatus
from cywl_oopz.commands.parsing import CommandTextParser
from cywl_oopz.commands.router import CommandRouter, ParsedCommand
from cywl_oopz.integrations.oopz.command_requests import OopzCommandRequestFactory


class CountingCommandTextParser(CommandTextParser):
    def __init__(self, prefix: str) -> None:
        super().__init__(prefix)
        self.calls = 0

    def parse(self, text: str):
        self.calls += 1
        return super().parse(text)


@dataclass
class FakeMessage:
    text: str
    plain_text: str = ""
    content: str = ""
    sender_id: str = "actor"
    area: str = "area"
    channel: str = "channel"
    message_id: str = "message"
    client_message_id: str = "client-message"
    timestamp: str = "123"
    reference_message_id: str = ""
    mention_list: list[object] = field(default_factory=list)


class FakeContext:
    def __init__(self, message: FakeMessage, *, private: bool = False) -> None:
        self.event = SimpleNamespace(message=message, is_private=private)
        self.replies: list[str] = []
        self.sent: list[str] = []
        self.reactions: list[str] = []

    async def reply(self, text: str) -> None:
        self.replies.append(text)

    async def send(self, text: str) -> None:
        self.sent.append(text)

    async def react(self, emoji: str) -> None:
        self.reactions.append(emoji)


class EchoCommand:
    name = "echo"
    description = "Echo input."

    def __init__(self) -> None:
        self.received: ParsedCommand | None = None

    async def execute(self, command: ParsedCommand, _context: object) -> None:
        self.received = command


def test_oopz_factory_projects_raw_text_mentions_reference_and_channel_location() -> None:
    target = "mentioned-person"
    message = FakeMessage(
        "/echo left(met)mentioned-person(met)right",
        plain_text="/echo leftright",
        reference_message_id="referenced-message",
        mention_list=[SimpleNamespace(person=target, is_bot=False, bot_type="")],
    )
    context = FakeContext(message)

    request = OopzCommandRequestFactory(CommandTextParser("/")).from_message(message, context)

    assert request is not None
    assert request.text is not None
    assert request.text.raw_tail == "left(met)mentioned-person(met)right"
    assert request.text.tokens == ("left(met)mentioned-person(met)right",)
    assert request.actor.person_id == "actor"
    assert request.location.scope is CommandScope.CHANNEL
    assert request.location.area_id == "area"
    assert request.location.channel_id == "channel"
    assert request.source.message_id == "message"
    assert request.target is not None
    assert request.target.message_id == "referenced-message"
    assert tuple(mention.person_id for mention in request.mentions) == (target,)


def test_oopz_factory_projects_private_location_without_area() -> None:
    message = FakeMessage("/echo hi", area="", channel="private-channel")

    request = OopzCommandRequestFactory(CommandTextParser("/")).from_message(
        message,
        FakeContext(message, private=True),
    )

    assert request is not None
    assert request.location.scope is CommandScope.PRIVATE
    assert request.location.area_id == ""
    assert request.location.channel_id == "private-channel"
    assert request.location.target_person_id == "actor"


def test_oopz_factory_ignores_non_command_before_validating_metadata() -> None:
    message = FakeMessage("hello", sender_id="", area="", channel="")

    request = OopzCommandRequestFactory(CommandTextParser("/")).from_message(
        message,
        FakeContext(message),
    )

    assert request is None


@pytest.mark.asyncio
async def test_dispatch_request_uses_the_factory_parse_exactly_once() -> None:
    parser = CountingCommandTextParser("/")
    router = CommandRouter("/", parser=parser)
    command = EchoCommand()
    router.register(command)
    message = FakeMessage("/echo hello  world")
    context = FakeContext(message)
    request = OopzCommandRequestFactory(parser).from_message(message, context)

    assert request is not None
    outcome = await router.dispatch_request(request, context)

    assert parser.calls == 1
    assert outcome.status is DispatchStatus.COMPLETED
    assert outcome.command_name == "echo"
    assert outcome.consumed is True
    assert command.received is not None
    assert command.received.arguments == ("hello", "world")
    assert command.received.raw_arguments == "hello  world"


@pytest.mark.asyncio
async def test_unknown_prefixed_request_is_explicitly_consumed() -> None:
    parser = CommandTextParser("/")
    router = CommandRouter("/", parser=parser)
    message = FakeMessage("/missing")
    context = FakeContext(message)
    request = OopzCommandRequestFactory(parser).from_message(message, context)

    assert request is not None
    outcome = await router.dispatch_request(request, context)

    assert outcome.status is DispatchStatus.UNKNOWN
    assert outcome.command_name == "missing"
    assert outcome.consumed is True
