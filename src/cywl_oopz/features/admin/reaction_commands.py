"""RBAC-protected administration actions triggered by message reactions."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from cywl_oopz.core.errors import AuthorizationError, DatabaseError
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.access.models import AccessPrincipal, AccessResource, Permission
from cywl_oopz.features.access.service import AuthorizationService

from .models import MessageRecallOutcome, OopzMessageAddress
from .ports import AgentDiagnosticRenderer, AgentDiagnosticRepository
from .recall import (
    BotMessageRecallTransportError,
    MessageRecallService,
    ReferencedBotMessageNotFoundError,
)
from .references import ReferencedMessageResolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReactionCommandInvocation:
    """Trusted reaction actor and exact reacted-message address."""

    emoji: str
    message_id: str
    principal: AccessPrincipal
    resource: AccessResource
    address: OopzMessageAddress

    def __post_init__(self) -> None:
        emoji = self.emoji.strip()
        message_id = self.message_id.strip()
        if not emoji or not message_id:
            raise ValueError("Reaction command requires an emoji and message ID")
        if len(message_id) > 256:
            raise ValueError("Reaction message ID must be at most 256 characters")
        object.__setattr__(self, "emoji", emoji)
        object.__setattr__(self, "message_id", message_id)


class ReactionCommandResponder(Protocol):
    """Deliver a tracked response beside the reacted message."""

    async def send(self, invocation: ReactionCommandInvocation, text: str) -> None:
        """Send one bounded response page."""


@dataclass(frozen=True, slots=True)
class ReactionCommandResult:
    """Zero or more messages emitted after a reaction action completes."""

    messages: tuple[str, ...] = ()


class ReactionCommandAction(Protocol):
    """One emoji-bound privileged action."""

    emoji: str
    permission: Permission

    async def execute(
        self,
        invocation: ReactionCommandInvocation,
    ) -> ReactionCommandResult:
        """Run the action after authorization succeeds."""


class ReactionCommandFailure(RuntimeError):
    """Expected reaction action failure with a user-safe message."""

    def __init__(self, user_message: str, *, code: str) -> None:
        self.user_message = user_message
        self.code = code
        super().__init__(code)


class ReactionCommandRouter:
    """Dispatch a curated emoji map with fresh RBAC checks."""

    target_timeout_seconds = 10.0

    def __init__(
        self,
        authorizer: AuthorizationService,
        targets: ReferencedMessageResolver,
        responder: ReactionCommandResponder,
    ) -> None:
        self._authorizer = authorizer
        self._targets = targets
        self._responder = responder
        self._actions: dict[str, ReactionCommandAction] = {}

    def register(self, action: ReactionCommandAction) -> None:
        emoji = action.emoji.strip()
        if not emoji:
            raise ValueError("Reaction command emoji must not be empty")
        if emoji in self._actions:
            raise ValueError(f"Reaction command already registered: {emoji}")
        self._actions[emoji] = action

    async def dispatch(self, invocation: ReactionCommandInvocation) -> bool:
        action = self._actions.get(invocation.emoji)
        if action is None:
            return False
        try:
            async with asyncio.timeout(self.target_timeout_seconds):
                target = await self._targets.resolve(
                    invocation.message_id,
                    invocation.address,
                    None,
                )
        except (BotMessageRecallTransportError, DatabaseError, TimeoutError) as exc:
            logger.warning(
                "Reaction command target lookup unavailable: message=%s error=%s",
                opaque_ref(invocation.message_id),
                exception_kind(exc),
            )
            return True
        if target is None:
            logger.debug(
                "Ignored reaction command on untracked message: emoji=%s message=%s",
                invocation.emoji,
                opaque_ref(invocation.message_id),
            )
            return True
        try:
            await self._authorizer.require(
                invocation.principal,
                action.permission,
                invocation.resource,
            )
        except AuthorizationError:
            logger.info(
                "Denied reaction command: emoji=%s principal=%s message=%s",
                invocation.emoji,
                opaque_ref(invocation.principal.person_id),
                opaque_ref(invocation.message_id),
            )
            await self._responder.send(invocation, "你没有执行此操作的权限。")
            return True
        except DatabaseError as exc:
            logger.warning(
                "Reaction command authorization unavailable: emoji=%s error=%s",
                invocation.emoji,
                exception_kind(exc),
            )
            await self._responder.send(invocation, "权限服务暂时不可用，请稍后重试。")
            return True

        try:
            result = await action.execute(invocation)
        except ReactionCommandFailure as exc:
            logger.info(
                "Reaction command rejected: emoji=%s message=%s code=%s",
                invocation.emoji,
                opaque_ref(invocation.message_id),
                exc.code,
            )
            await self._responder.send(invocation, exc.user_message)
            return True
        except Exception as exc:
            logger.exception(
                "Reaction command failed unexpectedly: emoji=%s message=%s error=%s",
                invocation.emoji,
                opaque_ref(invocation.message_id),
                exception_kind(exc),
            )
            await self._responder.send(invocation, "操作失败，请稍后重试。")
            return True

        for message in result.messages:
            await self._responder.send(invocation, message)
        logger.info(
            "Reaction command completed: emoji=%s principal=%s message=%s responses=%s",
            invocation.emoji,
            opaque_ref(invocation.principal.person_id),
            opaque_ref(invocation.message_id),
            len(result.messages),
        )
        return True


class RecallReactionCommand:
    """Recall the reacted Bot-owned message when 🫥 is added."""

    emoji = "🫥"
    permission = Permission.BOT_MESSAGE_RECALL
    timeout_seconds = 10.0

    def __init__(self, service: MessageRecallService) -> None:
        self._service = service

    async def execute(
        self,
        invocation: ReactionCommandInvocation,
    ) -> ReactionCommandResult:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                outcome = await self._service.recall(
                    invocation.message_id,
                    invocation.address,
                )
        except ReferencedBotMessageNotFoundError:
            logger.debug(
                "Reaction recall target disappeared: message=%s",
                opaque_ref(invocation.message_id),
            )
            return ReactionCommandResult()
        except (BotMessageRecallTransportError, TimeoutError) as exc:
            raise ReactionCommandFailure(
                "撤回失败，请稍后重试。",
                code="recall_unavailable",
            ) from exc
        except DatabaseError as exc:
            raise ReactionCommandFailure(
                "撤回服务暂时不可用，请稍后重试。",
                code="recall_persistence_unavailable",
            ) from exc
        if outcome is MessageRecallOutcome.ALREADY_RECALLED:
            raise ReactionCommandFailure(
                "这条回复已经撤回。",
                code="already_recalled",
            )
        return ReactionCommandResult()


class DebugReactionCommand:
    """Render normal bounded Agent diagnostics when 🤯 is added."""

    emoji = "🤯"
    permission = Permission.AGENT_RESPONSE_DEBUG
    timeout_seconds = 10.0

    def __init__(
        self,
        repository: AgentDiagnosticRepository,
        renderer: AgentDiagnosticRenderer,
    ) -> None:
        self._repository = repository
        self._renderer = renderer

    async def execute(
        self,
        invocation: ReactionCommandInvocation,
    ) -> ReactionCommandResult:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                diagnostic = await self._repository.get_by_outbound_message(
                    invocation.message_id,
                    invocation.address,
                )
        except (DatabaseError, TimeoutError) as exc:
            raise ReactionCommandFailure(
                "诊断服务暂时不可用，请稍后重试。",
                code="diagnostic_unavailable",
            ) from exc
        if diagnostic is None:
            logger.debug(
                "Ignored debug reaction without Agent diagnostic: message=%s",
                opaque_ref(invocation.message_id),
            )
            return ReactionCommandResult()
        return ReactionCommandResult(self._renderer.render(diagnostic, verbose=False))
