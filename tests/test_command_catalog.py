from __future__ import annotations

from dataclasses import dataclass

import pytest

from cywl_oopz.commands.catalog import CommandCatalog, CommandSpec
from cywl_oopz.commands.definitions import CommandDefinition, PublicCommandAuthorization
from cywl_oopz.commands.models import CommandRequest, CommandText
from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.testing.commands import dispatch_command


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
    sender_id: str = "person"
    area: str = "area"
    channel: str = "channel"


class AliasCommand:
    name = "echo"
    description = "Echo."
    category = "test"
    usage = ("echo <text>",)
    aliases = ("say",)

    def __init__(self) -> None:
        self.command: CommandText | None = None

    def definition(self) -> CommandDefinition[CommandText]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            self,
            self,
            PublicCommandAuthorization(),
        )

    def parse(self, request: CommandRequest) -> CommandText:
        assert request.text is not None
        return request.text

    async def handle(self, request: CommandRequest, command: CommandText) -> None:
        del request
        self.command = command


@pytest.mark.asyncio
async def test_router_dispatches_alias_to_the_canonical_definition() -> None:
    router = CommandRouter("/")
    command = AliasCommand()
    router.register_definition(command.definition())

    consumed = await dispatch_command(router, FakeMessage("/say hello"), object())

    assert consumed is True
    assert command.command is not None
    assert command.command.name == "say"
    assert command.command.tokens == ("hello",)
    assert router.specs == (
        CommandSpec(
            "echo",
            "Echo.",
            "test",
            ("echo <text>",),
            aliases=("say",),
        ),
    )
