"""Asynchronous, object-oriented command dispatch."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, Protocol

from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from cywl_oopz.core.errors import AuthorizationError, DatabaseError
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.access.models import AccessPrincipal, AccessResource
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.integrations.oopz.access import OopzAccessInvocation

from .catalog import CommandCatalog, CommandSpec
from .definitions import (
    AccessRequirement,
    CommandDefinition,
    CommandExecutionPolicy,
    CommandUsageError,
    ExecutionMode,
)
from .execution import CommandTaskSupervisor
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
    """A command line used only by handlers not yet migrated to typed definitions."""

    name: str
    arguments: tuple[str, ...]
    raw_arguments: str = field(default="", compare=False)

    @classmethod
    def from_text(cls, text: CommandText) -> ParsedCommand:
        """Project new root text into the temporary legacy command contract."""
        return cls(text.name, text.tokens, text.raw_tail)


class Command(Protocol):
    """Temporary contract for command implementations awaiting migration."""

    name: str
    description: str

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        """Handle one parsed command through the legacy SDK context."""


class CommandAccessPolicy(Protocol):
    """Temporary parsed-command authorization contract."""

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
    """Exactly one legacy command or typed definition stored in the catalog."""

    spec: CommandSpec
    command: Command | None = None
    access: CommandAccessPolicy | None = None
    definition: CommandDefinition[Any] | None = None
    execution: CommandExecutionPolicy = field(default_factory=CommandExecutionPolicy)

    def __post_init__(self) -> None:
        if (self.command is None) == (self.definition is None):
            raise ValueError("Registration requires exactly one command implementation")

    @property
    def implementation(self) -> Any:
        if self.command is not None:
            return self.command
        assert self.definition is not None
        return self.definition.handler


class CommandRouter:
    """Catalog and dispatcher supporting typed definitions during migration."""

    def __init__(
        self,
        prefix: str,
        authorizer: AuthorizationService | None = None,
        *,
        parser: CommandTextParser | None = None,
        supervisor: CommandTaskSupervisor | None = None,
    ) -> None:
        self._parser = parser or CommandTextParser(prefix)
        if self._parser.prefix != prefix:
            raise ValueError("Command router and parser prefixes must match")
        self.prefix = self._parser.prefix
        self._authorizer = authorizer
        self._supervisor = supervisor
        self._catalog: CommandCatalog[RegisteredCommand] = CommandCatalog()

    @property
    def commands(self) -> tuple[Any, ...]:
        """Registered implementations in stable, alphabetical order."""
        return tuple(registration.implementation for registration in self._catalog.entries)

    @property
    def specs(self) -> tuple[CommandSpec, ...]:
        """Registered root metadata in stable, alphabetical order."""
        return tuple(registration.spec for registration in self._catalog.entries)

    def register(
        self,
        command: Command,
        *,
        access: CommandAccessPolicy | None = None,
        spec: CommandSpec | None = None,
        execution: CommandExecutionPolicy | None = None,
    ) -> None:
        """Register one legacy command behind the compatibility adapter."""
        resolved_spec = spec or CommandSpec.from_command(command)
        if resolved_spec.name != command.name.strip().casefold():
            raise ValueError("Command spec name must match the implementation name")
        if access is not None and self._authorizer is None:
            raise ValueError("Restricted commands require a CommandRouter authorizer")
        self._catalog.register(
            resolved_spec,
            RegisteredCommand(
                resolved_spec,
                command=command,
                access=access,
                execution=execution or CommandExecutionPolicy(),
            ),
        )

    def register_definition(self, definition: CommandDefinition[Any]) -> None:
        """Register a complete typed parser/authorization/handler definition."""
        self._catalog.register(
            definition.spec,
            RegisteredCommand(definition.spec, definition=definition),
        )

    async def available_commands(self, context: EventContext) -> tuple[Any, ...]:
        """Compatibility discovery entry point for legacy Help handlers."""
        registrations = await self._available_registrations(context)
        return tuple(registration.implementation for registration in registrations)

    async def available_specs(self, context: EventContext) -> tuple[CommandSpec, ...]:
        """Compatibility discovery entry point for legacy Help handlers."""
        registrations = await self._available_registrations(context)
        return tuple(
            registration.spec for registration in registrations if not registration.spec.hidden
        )

    async def available_specs_for(self, request: CommandRequest) -> tuple[CommandSpec, ...]:
        """Discover commands without parsing or reading the platform event again."""
        registrations = await self._available_registrations_for(request)
        return tuple(
            registration.spec for registration in registrations if not registration.spec.hidden
        )

    async def _available_registrations(
        self,
        context: EventContext,
    ) -> tuple[RegisteredCommand, ...]:
        from cywl_oopz.integrations.oopz.command_requests import OopzCommandRequestFactory

        request = OopzCommandRequestFactory(self._parser).for_visibility(context)
        return await self._available_registrations_for(request)

    async def _available_registrations_for(
        self,
        request: CommandRequest,
    ) -> tuple[RegisteredCommand, ...]:
        invocation = self._access_invocation(request)
        available: list[RegisteredCommand] = []
        for registration in self._catalog.entries:
            definition = registration.definition
            if definition is not None:
                authorization = definition.authorization
                if not authorization.is_available(request):
                    continue
                requirement = authorization.visibility_requirement(request)
            else:
                access = registration.access
                if access is None:
                    available.append(registration)
                    continue
                if not access.is_available(invocation):
                    continue
                requirement = access.visibility_requirement(invocation)
            if await self._is_visible(registration.spec.name, invocation, requirement):
                available.append(registration)
        return tuple(available)

    async def _is_visible(
        self,
        command_name: str,
        invocation: OopzAccessInvocation,
        requirement: AccessRequirement | None,
    ) -> bool:
        if requirement is None:
            return True
        if self._authorizer is None:
            raise ValueError(f"Restricted command requires an authorizer: {command_name}")
        try:
            return await self._authorizer.allows(
                invocation.principal,
                requirement.permission,
                requirement.resource,
            )
        except DatabaseError as exc:
            logger.warning(
                "Could not resolve restricted command visibility: command=%s error=%s",
                command_name,
                type(exc).__name__,
            )
            return False

    async def dispatch(self, message: OopzMessage, context: EventContext) -> bool:
        """Compatibility entry point for legacy Feature tests and integrations."""
        command = self.parse_message(message)
        if command is None:
            return False
        registration = self._catalog.get(command.name)
        if registration is None:
            return False
        if registration.definition is not None:
            from cywl_oopz.integrations.oopz.command_requests import OopzCommandRequestFactory

            request = OopzCommandRequestFactory(self._parser).from_message(message, context)
            assert request is not None
            outcome = await self.dispatch_request(request, context)
        else:
            command = self._canonical(command, registration.spec)
            invocation = (
                OopzAccessInvocation.from_context(context)
                if registration.access is not None
                else None
            )
            outcome = await self._execute_legacy(
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
        registration = self._catalog.get(text.name)
        if registration is None:
            return DispatchOutcome(DispatchStatus.UNKNOWN, text.name)
        if registration.definition is not None:
            return await self._execute_definition(registration.definition, request)

        command = self._canonical(ParsedCommand.from_text(text), registration.spec)
        invocation = self._access_invocation(request) if registration.access is not None else None
        return await self._schedule_or_run(
            command.name,
            registration.execution,
            request,
            self._run_legacy(
                registration,
                command,
                invocation,
                request,
                legacy_context,
            ),
        )

    async def _execute_definition(
        self,
        definition: CommandDefinition[Any],
        request: CommandRequest,
    ) -> DispatchOutcome:
        try:
            arguments = definition.parser.parse(request)
        except CommandUsageError as exc:
            await request.responder.reply(exc.render(definition.spec, self.prefix))
            return DispatchOutcome(DispatchStatus.COMPLETED, definition.spec.name)

        return await self._schedule_or_run(
            definition.spec.name,
            definition.execution,
            request,
            self._run_definition(definition, request, arguments),
        )

    async def _schedule_or_run(
        self,
        command_name: str,
        execution: CommandExecutionPolicy,
        request: CommandRequest,
        operation: Coroutine[Any, Any, DispatchOutcome],
    ) -> DispatchOutcome:
        if execution.mode is ExecutionMode.BACKGROUND and self._supervisor is not None:
            request_ref = self._request_ref(request)
            if self._supervisor.start(command_name, request_ref, operation):
                return DispatchOutcome(DispatchStatus.STARTED, command_name)
            await self._safe_reply(
                request.responder,
                "Bot 正在关闭，暂不接受新的命令。",
                command_name,
            )
            return DispatchOutcome(DispatchStatus.IGNORED, command_name)
        return await operation

    async def _run_definition(
        self,
        definition: CommandDefinition[Any],
        request: CommandRequest,
        arguments: Any,
    ) -> DispatchOutcome:
        started_at = time.monotonic()
        command_name = definition.spec.name
        request_ref = self._request_ref(request)
        outcome: DispatchOutcome | None = None
        logger.debug(
            "Command execution started: command=%s trigger=%s request=%s",
            command_name,
            request.trigger.value,
            request_ref,
        )
        try:
            if definition.execution.timeout_seconds is None:
                outcome = await self._authorize_and_handle(definition, request, arguments)
            else:
                async with asyncio.timeout(definition.execution.timeout_seconds):
                    outcome = await self._authorize_and_handle(definition, request, arguments)
            return outcome
        except TimeoutError as exc:
            timeout_seconds = definition.execution.timeout_seconds
            if timeout_seconds is None:
                logger.exception(
                    "Command execution failed unexpectedly: command=%s request=%s error=%s",
                    command_name,
                    request_ref,
                    exception_kind(exc),
                )
                user_message = "命令执行失败，请稍后重试。"
            else:
                logger.warning(
                    "Command execution timed out: command=%s request=%s timeout_seconds=%.1f",
                    command_name,
                    request_ref,
                    timeout_seconds,
                )
                user_message = "命令执行超时，请稍后重试。"
            await self._safe_reply(
                request.responder,
                user_message,
                command_name,
            )
            outcome = DispatchOutcome(DispatchStatus.FAILED, command_name)
            return outcome
        except asyncio.CancelledError:
            logger.info(
                "Command execution cancelled: command=%s request=%s",
                command_name,
                request_ref,
            )
            raise
        except Exception as exc:
            logger.exception(
                "Command execution failed unexpectedly: command=%s request=%s error=%s",
                command_name,
                request_ref,
                exception_kind(exc),
            )
            await self._safe_reply(
                request.responder,
                "命令执行失败，请稍后重试。",
                command_name,
            )
            outcome = DispatchOutcome(DispatchStatus.FAILED, command_name)
            return outcome
        finally:
            logger.info(
                "Command execution finished: command=%s request=%s outcome=%s duration_ms=%s",
                command_name,
                request_ref,
                outcome.status.value if outcome is not None else "cancelled",
                round((time.monotonic() - started_at) * 1000),
            )

    async def _authorize_and_handle(
        self,
        definition: CommandDefinition[Any],
        request: CommandRequest,
        arguments: Any,
    ) -> DispatchOutcome:
        requirement = definition.authorization.requirement(request, arguments)
        if requirement is not None:
            outcome = await self._require(
                definition.spec.name,
                self._access_invocation(request),
                requirement,
                request.responder,
            )
            if outcome is not None:
                return outcome

        await definition.handler.handle(request, arguments)
        return DispatchOutcome(DispatchStatus.COMPLETED, definition.spec.name)

    async def _execute_legacy(
        self,
        registration: RegisteredCommand,
        command: ParsedCommand,
        invocation: OopzAccessInvocation | None,
        responder: CommandResponder,
        legacy_context: Any,
    ) -> DispatchOutcome:
        access = registration.access
        if access is not None:
            assert invocation is not None
            requirement = access.requirement(command, invocation)
            if requirement is not None:
                outcome = await self._require(command.name, invocation, requirement, responder)
                if outcome is not None:
                    return outcome

        assert registration.command is not None
        try:
            await registration.command.execute(command, legacy_context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Legacy command failed unexpectedly: command=%s error=%s",
                command.name,
                exception_kind(exc),
            )
            await self._safe_reply(
                responder,
                "命令执行失败，请稍后重试。",
                command.name,
            )
            return DispatchOutcome(DispatchStatus.FAILED, command.name)
        return DispatchOutcome(DispatchStatus.COMPLETED, command.name)

    async def _run_legacy(
        self,
        registration: RegisteredCommand,
        command: ParsedCommand,
        invocation: OopzAccessInvocation | None,
        request: CommandRequest,
        legacy_context: Any,
    ) -> DispatchOutcome:
        started_at = time.monotonic()
        request_ref = self._request_ref(request)
        outcome: DispatchOutcome | None = None
        logger.debug(
            "Legacy command execution started: command=%s request=%s",
            command.name,
            request_ref,
        )
        try:
            if registration.execution.timeout_seconds is None:
                outcome = await self._execute_legacy(
                    registration,
                    command,
                    invocation,
                    request.responder,
                    legacy_context,
                )
            else:
                async with asyncio.timeout(registration.execution.timeout_seconds):
                    outcome = await self._execute_legacy(
                        registration,
                        command,
                        invocation,
                        request.responder,
                        legacy_context,
                    )
            return outcome
        except TimeoutError:
            logger.warning(
                "Legacy command execution timed out: command=%s request=%s timeout_seconds=%.1f",
                command.name,
                request_ref,
                registration.execution.timeout_seconds,
            )
            await self._safe_reply(
                request.responder,
                "命令执行超时，请稍后重试。",
                command.name,
            )
            outcome = DispatchOutcome(DispatchStatus.FAILED, command.name)
            return outcome
        except asyncio.CancelledError:
            logger.info(
                "Legacy command execution cancelled: command=%s request=%s",
                command.name,
                request_ref,
            )
            raise
        finally:
            logger.info(
                "Legacy command execution finished: command=%s request=%s outcome=%s "
                "duration_ms=%s",
                command.name,
                request_ref,
                outcome.status.value if outcome is not None else "cancelled",
                round((time.monotonic() - started_at) * 1000),
            )

    async def _require(
        self,
        command_name: str,
        invocation: OopzAccessInvocation,
        requirement: AccessRequirement,
        responder: CommandResponder,
    ) -> DispatchOutcome | None:
        if self._authorizer is None:
            raise ValueError(f"Restricted command requires an authorizer: {command_name}")
        try:
            await self._authorizer.require(
                invocation.principal,
                requirement.permission,
                requirement.resource,
            )
        except AuthorizationError:
            logger.info(
                "Denied command: command=%s principal=%s permission=%s resource=%s",
                command_name,
                opaque_ref(invocation.principal.person_id),
                requirement.permission.value,
                self._resource_ref(requirement.resource),
            )
            await responder.reply("你没有执行此操作的权限。")
            return DispatchOutcome(DispatchStatus.DENIED, command_name)
        except DatabaseError as exc:
            logger.warning(
                "Command authorization unavailable: command=%s error=%s",
                command_name,
                type(exc).__name__,
            )
            await responder.reply("权限服务暂时不可用，请稍后重试。")
            return DispatchOutcome(DispatchStatus.FAILED, command_name)
        return None

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
    def _canonical(command: ParsedCommand, spec: CommandSpec) -> ParsedCommand:
        if command.name == spec.name:
            return command
        return ParsedCommand(spec.name, command.arguments, command.raw_arguments)

    @staticmethod
    def _resource_ref(resource: AccessResource) -> str:
        return opaque_ref(resource.kind.value, resource.area_id, resource.channel_id)

    @staticmethod
    def _request_ref(request: CommandRequest) -> str:
        return opaque_ref(
            request.trigger.value,
            request.source.message_id,
            request.actor.person_id,
            request.location.scope.value,
            request.location.area_id,
            request.location.channel_id,
        )

    @staticmethod
    async def _safe_reply(
        responder: CommandResponder,
        message: str,
        command_name: str,
    ) -> None:
        try:
            await responder.reply(message)
        except Exception as exc:
            logger.warning(
                "Could not deliver command failure response: command=%s error=%s",
                command_name,
                exception_kind(exc),
            )
