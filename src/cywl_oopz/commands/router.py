"""Asynchronous, object-oriented command dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """A command line after its prefix and command name have been parsed."""

    name: str
    arguments: tuple[str, ...]


class Command(Protocol):
    """Contract for a command implementation."""

    name: str
    description: str

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        """Handle one parsed command."""


class CommandRouter:
    """Maps prefixed chat messages to asynchronous command objects."""

    def __init__(self, prefix: str) -> None:
        if not prefix:
            raise ValueError("Command prefix must not be empty")
        self.prefix = prefix
        self._commands: dict[str, Command] = {}

    @property
    def commands(self) -> tuple[Command, ...]:
        """Registered commands in stable, alphabetical order."""
        return tuple(self._commands[name] for name in sorted(self._commands))

    def register(self, command: Command) -> None:
        """Register a unique command object."""
        name = command.name.casefold()
        if not name:
            raise ValueError("Command name must not be empty")
        if name in self._commands:
            raise ValueError(f"Command already registered: {command.name}")
        self._commands[name] = command

    async def dispatch(self, message: OopzMessage, context: EventContext) -> bool:
        """Execute a matching command and report whether the message was consumed."""
        command = self.parse(message.plain_text or message.text or message.content)
        if command is None:
            return False

        handler = self._commands.get(command.name)
        if handler is None:
            return False

        await handler.execute(command, context)
        return True

    def parse(self, text: str) -> ParsedCommand | None:
        """Parse one prefix command without interpreting non-command chat messages."""
        content = text.strip()
        if not content.startswith(self.prefix):
            return None

        parts = content[len(self.prefix) :].split()
        if not parts:
            return None
        return ParsedCommand(name=parts[0].casefold(), arguments=tuple(parts[1:]))
