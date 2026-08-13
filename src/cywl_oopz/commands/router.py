"""Framework-neutral asynchronous command dispatch."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from typing import Any

from cywl_oopz.core.errors import AuthorizationError, DatabaseError
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.access.models import AccessPrincipal, AccessResource
from cywl_oopz.features.access.service import AuthorizationService

from .catalog import CommandCatalog, CommandSpec
from .definitions import (
    AccessRequirement,
    CommandDefinition,
    CommandExecutionPolicy,
    CommandUsageError,
    ExecutionMode,
)
from .execution import CommandTaskSupervisor
from .models import CommandRequest, DispatchOutcome, DispatchStatus
from .parsing import CommandTextParser
from .responses import CommandResponder

logger = logging.getLogger(__name__)


class CommandRouter:
    """Catalog, authorize, and execute complete typed command definitions."""

    def __init__(
        self,
        prefix: str,
        authorizer: AuthorizationService | None = None,
        *,
        supervisor: CommandTaskSupervisor | None = None,
    ) -> None:
        self.prefix = CommandTextParser(prefix).prefix
        self._authorizer = authorizer
        self._supervisor = supervisor
        self._catalog: CommandCatalog[CommandDefinition[Any]] = CommandCatalog()

    @property
    def commands(self) -> tuple[Any, ...]:
        """Registered handlers in stable, alphabetical order."""
        return tuple(definition.handler for definition in self._catalog.entries)

    @property
    def specs(self) -> tuple[CommandSpec, ...]:
        """Registered root metadata in stable, alphabetical order."""
        return tuple(definition.spec for definition in self._catalog.entries)

    def register_definition(self, definition: CommandDefinition[Any]) -> None:
        """Register one complete parser/authorization/handler definition."""
        self._catalog.register(definition.spec, definition)

    async def available_specs_for(self, request: CommandRequest) -> tuple[CommandSpec, ...]:
        """Return request-visible commands without reading transport state again."""
        principal = AccessPrincipal(request.actor.person_id)
        available: list[CommandSpec] = []
        for definition in self._catalog.entries:
            authorization = definition.authorization
            if not authorization.is_available(request):
                continue
            requirement = authorization.visibility_requirement(request)
            if await self._is_visible(definition.spec.name, principal, requirement):
                if not definition.spec.hidden:
                    available.append(definition.spec)
        return tuple(available)

    async def _is_visible(
        self,
        command_name: str,
        principal: AccessPrincipal,
        requirement: AccessRequirement | None,
    ) -> bool:
        if requirement is None:
            return True
        if self._authorizer is None:
            raise ValueError(f"Restricted command requires an authorizer: {command_name}")
        try:
            return await self._authorizer.allows(
                principal,
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

    async def dispatch_request(self, request: CommandRequest) -> DispatchOutcome:
        """Dispatch one already-parsed request without transport dependencies."""
        text = request.text
        if text is None:
            return DispatchOutcome(DispatchStatus.NOT_A_COMMAND)
        definition = self._catalog.get(text.name)
        if definition is None:
            return DispatchOutcome(DispatchStatus.UNKNOWN, text.name)
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
            await self._safe_reply(request.responder, user_message, command_name)
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
                AccessPrincipal(request.actor.person_id),
                requirement,
                request.responder,
            )
            if outcome is not None:
                return outcome
        await definition.handler.handle(request, arguments)
        return DispatchOutcome(DispatchStatus.COMPLETED, definition.spec.name)

    async def _require(
        self,
        command_name: str,
        principal: AccessPrincipal,
        requirement: AccessRequirement,
        responder: CommandResponder,
    ) -> DispatchOutcome | None:
        if self._authorizer is None:
            raise ValueError(f"Restricted command requires an authorizer: {command_name}")
        try:
            await self._authorizer.require(
                principal,
                requirement.permission,
                requirement.resource,
            )
        except AuthorizationError:
            logger.info(
                "Denied command: command=%s principal=%s permission=%s resource=%s",
                command_name,
                opaque_ref(principal.person_id),
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
