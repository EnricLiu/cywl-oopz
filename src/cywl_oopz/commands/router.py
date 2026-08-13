"""Asynchronous, object-oriented command dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from cywl_oopz.core.errors import AuthorizationError, DatabaseError
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.access.models import AccessResource, Permission
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.integrations.oopz.access import OopzAccessInvocation

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """A command line after its prefix and command name have been parsed."""

    name: str
    arguments: tuple[str, ...]
    raw_arguments: str = field(default="", compare=False)


class Command(Protocol):
    """Contract for a command implementation."""

    name: str
    description: str

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        """Handle one parsed command."""


@dataclass(frozen=True, slots=True)
class AccessRequirement:
    """One concrete permission check resolved for a parsed command."""

    permission: Permission
    resource: AccessResource


class CommandAccessPolicy(Protocol):
    """Resolve subcommand-aware dispatch and help visibility requirements."""

    def is_available(self, invocation: OopzAccessInvocation) -> bool:
        """Return whether this command has a usable path in the current context."""

    def requirement(
        self,
        command: ParsedCommand,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement | None:
        """Return the check required for this invocation, or none for a public path."""

    def visibility_requirement(
        self,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement | None:
        """Return the check required to show this command in `/help`."""


@dataclass(frozen=True, slots=True)
class RegisteredCommand:
    """A command and its explicit optional authorization policy."""

    command: Command
    access: CommandAccessPolicy | None = None


class CommandRouter:
    """Maps prefixed chat messages to asynchronous command objects."""

    def __init__(
        self,
        prefix: str,
        authorizer: AuthorizationService | None = None,
    ) -> None:
        if not prefix:
            raise ValueError("Command prefix must not be empty")
        self.prefix = prefix
        self._authorizer = authorizer
        self._commands: dict[str, RegisteredCommand] = {}

    @property
    def commands(self) -> tuple[Command, ...]:
        """Registered commands in stable, alphabetical order."""
        return tuple(self._commands[name].command for name in sorted(self._commands))

    def register(
        self,
        command: Command,
        *,
        access: CommandAccessPolicy | None = None,
    ) -> None:
        """Register a unique command object."""
        name = command.name.casefold()
        if not name:
            raise ValueError("Command name must not be empty")
        if name in self._commands:
            raise ValueError(f"Command already registered: {command.name}")
        if access is not None and self._authorizer is None:
            raise ValueError("Restricted commands require a CommandRouter authorizer")
        self._commands[name] = RegisteredCommand(command, access)

    async def available_commands(self, context: EventContext) -> tuple[Command, ...]:
        """Return commands visible to this caller, preserving stable name order."""
        invocation: OopzAccessInvocation | None = None
        available: list[Command] = []
        for name in sorted(self._commands):
            registration = self._commands[name]
            access = registration.access
            if access is None:
                available.append(registration.command)
                continue
            if invocation is None:
                invocation = OopzAccessInvocation.from_context(context)
            if not access.is_available(invocation):
                continue
            requirement = access.visibility_requirement(invocation)
            if requirement is None:
                available.append(registration.command)
                continue
            assert self._authorizer is not None
            try:
                allowed = await self._authorizer.allows(
                    invocation.principal,
                    requirement.permission,
                    requirement.resource,
                )
            except DatabaseError as exc:
                logger.warning(
                    "Could not resolve restricted command visibility: command=%s error=%s",
                    name,
                    type(exc).__name__,
                )
                continue
            if allowed:
                available.append(registration.command)
        return tuple(available)

    async def dispatch(self, message: OopzMessage, context: EventContext) -> bool:
        """Execute a matching command and report whether the message was consumed."""
        command = self.parse(message.plain_text or message.text or message.content)
        if command is None:
            return False

        registration = self._commands.get(command.name)
        if registration is None:
            return False

        access = registration.access
        if access is not None:
            invocation = OopzAccessInvocation.from_context(context)
            requirement = access.requirement(command, invocation)
            if requirement is not None:
                assert self._authorizer is not None
                try:
                    await self._authorizer.require(
                        invocation.principal,
                        requirement.permission,
                        requirement.resource,
                    )
                except AuthorizationError:
                    logger.info(
                        "Denied command: command=%s principal=%s permission=%s resource=%s",
                        command.name,
                        opaque_ref(invocation.principal.person_id),
                        requirement.permission.value,
                        self._resource_ref(requirement.resource),
                    )
                    await context.reply("你没有执行此操作的权限。")
                    return True
                except DatabaseError as exc:
                    logger.warning(
                        "Command authorization unavailable: command=%s error=%s",
                        command.name,
                        type(exc).__name__,
                    )
                    await context.reply("权限服务暂时不可用，请稍后重试。")
                    return True

        await registration.command.execute(command, context)
        return True

    def parse(self, text: str) -> ParsedCommand | None:
        """Parse one prefix command without interpreting non-command chat messages."""
        content = text.strip()
        if not content.startswith(self.prefix):
            return None

        command_line = content[len(self.prefix) :].lstrip()
        parts = command_line.split(maxsplit=1)
        if not parts:
            return None
        raw_arguments = parts[1] if len(parts) == 2 else ""
        return ParsedCommand(
            name=parts[0].casefold(),
            arguments=tuple(raw_arguments.split()),
            raw_arguments=raw_arguments,
        )

    @staticmethod
    def _resource_ref(resource: AccessResource) -> str:
        return opaque_ref(resource.kind.value, resource.area_id, resource.channel_id)
