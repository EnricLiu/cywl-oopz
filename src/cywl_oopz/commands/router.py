"""Asynchronous, object-oriented command dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from cywl_oopz.core.errors import AuthorizationError, DatabaseError
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.access.models import AccessPrincipal, AccessResource, Permission
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.integrations.oopz.access import OopzAccessInvocation

from .models import (
    CommandRequest,
    CommandScope,
    CommandText,
    DispatchOutcome,
    DispatchStatus,
)
from .parsing import CommandTextParser
from .responses import CommandResponder

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """A command line after its prefix and command name have been parsed."""

    name: str
    arguments: tuple[str, ...]
    raw_arguments: str = field(default="", compare=False)

    @classmethod
    def from_text(cls, text: CommandText) -> ParsedCommand:
        """Project new root text into the temporary legacy command contract."""
        return cls(text.name, text.tokens, text.raw_tail)


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
        *,
        parser: CommandTextParser | None = None,
    ) -> None:
        self._parser = parser or CommandTextParser(prefix)
        if self._parser.prefix != prefix:
            raise ValueError("Command router and parser prefixes must match")
        self.prefix = self._parser.prefix
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
        """Compatibility entry point for legacy feature tests and integrations."""
        command = self.parse_message(message)
        if command is None:
            return False

        registration = self._commands.get(command.name)
        if registration is None:
            return False

        invocation = (
            OopzAccessInvocation.from_context(context) if registration.access is not None else None
        )
        outcome = await self._execute(
            registration,
            command,
            invocation,
            context,
            context,
        )
        return outcome.status not in {DispatchStatus.NOT_A_COMMAND, DispatchStatus.UNKNOWN}

    async def dispatch_request(
        self,
        request: CommandRequest,
        legacy_context: Any,
    ) -> DispatchOutcome:
        """Dispatch one already-parsed request without reading the SDK message again."""
        text = request.text
        if text is None:
            return DispatchOutcome(DispatchStatus.NOT_A_COMMAND)
        registration = self._commands.get(text.name)
        if registration is None:
            return DispatchOutcome(DispatchStatus.UNKNOWN, text.name)

        invocation = self._access_invocation(request) if registration.access is not None else None
        return await self._execute(
            registration,
            ParsedCommand.from_text(text),
            invocation,
            request.responder,
            legacy_context,
        )

    async def _execute(
        self,
        registration: RegisteredCommand,
        command: ParsedCommand,
        invocation: OopzAccessInvocation | None,
        responder: CommandResponder,
        legacy_context: Any,
    ) -> DispatchOutcome:
        """Authorize and invoke the legacy command behind the new request boundary."""

        access = registration.access
        if access is not None:
            assert invocation is not None
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
                    await responder.reply("你没有执行此操作的权限。")
                    return DispatchOutcome(DispatchStatus.DENIED, command.name)
                except DatabaseError as exc:
                    logger.warning(
                        "Command authorization unavailable: command=%s error=%s",
                        command.name,
                        type(exc).__name__,
                    )
                    await responder.reply("权限服务暂时不可用，请稍后重试。")
                    return DispatchOutcome(DispatchStatus.FAILED, command.name)

        await registration.command.execute(command, legacy_context)
        return DispatchOutcome(DispatchStatus.COMPLETED, command.name)

    def parse_message(self, message: OopzMessage) -> ParsedCommand | None:
        """Parse raw OOPZ text before its mention segments are removed from plain text."""
        text = str(
            getattr(message, "text", "")
            or getattr(message, "content", "")
            or getattr(message, "plain_text", "")
        )
        return self.parse(text)

    def parse(self, text: str) -> ParsedCommand | None:
        """Compatibility projection over the project-owned root parser."""
        parsed = self._parser.parse(text)
        return ParsedCommand.from_text(parsed) if parsed is not None else None

    @staticmethod
    def _access_invocation(request: CommandRequest) -> OopzAccessInvocation:
        resource = (
            AccessResource.private()
            if request.location.scope is CommandScope.PRIVATE
            else AccessResource.channel(
                request.location.area_id,
                request.location.channel_id,
            )
        )
        return OopzAccessInvocation(AccessPrincipal(request.actor.person_id), resource)

    @staticmethod
    def _resource_ref(resource: AccessResource) -> str:
        return opaque_ref(resource.kind.value, resource.area_id, resource.channel_id)
