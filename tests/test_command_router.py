from dataclasses import dataclass

import pytest

from cywl_oopz.commands.router import CommandRouter, ParsedCommand


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


def test_parse_requires_prefix() -> None:
    router = CommandRouter("!")

    assert router.parse("hello") is None
    assert router.parse("!") is None
    assert router.parse("!ECHO one two") == ParsedCommand("echo", ("one", "two"))


@pytest.mark.asyncio
async def test_dispatch_executes_registered_command() -> None:
    router = CommandRouter("!")
    command = EchoCommand()
    router.register(command)

    consumed = await router.dispatch(FakeMessage("!echo hello"), object())

    assert consumed is True
    assert command.received == ParsedCommand("echo", ("hello",))
