from __future__ import annotations

from dataclasses import dataclass

import pytest

from cywl_oopz.commands.catalog import CommandCatalog, CommandSpec
from cywl_oopz.commands.router import CommandRouter, ParsedCommand


def spec(name: str, *, aliases: tuple[str, ...] = ()) -> CommandSpec:
    return CommandSpec(
        name,
        f"{name} summary",
        "test",
        (name,),
        aliases=aliases,
    )


def test_catalog_resolves_alias_to_the_canonical_entry() -> None:
    catalog: CommandCatalog[str] = CommandCatalog()
    catalog.register(spec("echo", aliases=("say",)), "entry")

    assert catalog.get("echo") == "entry"
    assert catalog.get("SAY") == "entry"


@pytest.mark.parametrize("conflicting_name", ["echo", "say"])
def test_catalog_rejects_name_or_alias_collisions(conflicting_name: str) -> None:
    catalog: CommandCatalog[str] = CommandCatalog()
    catalog.register(spec("echo", aliases=("say",)), "first")

    with pytest.raises(ValueError, match="already registered"):
        catalog.register(spec(conflicting_name), "second")


@dataclass
class FakeMessage:
    plain_text: str
    text: str = ""
    content: str = ""


class AliasCommand:
    name = "echo"
    description = "Echo."
    category = "test"
    usage = ("echo <text>",)
    aliases = ("say",)

    def __init__(self) -> None:
        self.command: ParsedCommand | None = None

    async def execute(self, command: ParsedCommand, _context: object) -> None:
        self.command = command


@pytest.mark.asyncio
async def test_router_dispatches_alias_with_canonical_command_name() -> None:
    router = CommandRouter("/")
    command = AliasCommand()
    router.register(command)

    consumed = await router.dispatch(FakeMessage("/say hello"), object())

    assert consumed is True
    assert command.command == ParsedCommand("echo", ("hello",))
    assert router.specs == (
        CommandSpec(
            "echo",
            "Echo.",
            "test",
            ("echo <text>",),
            aliases=("say",),
        ),
    )
