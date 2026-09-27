"""Shared response lifecycle for command and event chat entry points."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from cywl_oopz.core.observability import opaque_ref

from .error_presenter import ChatErrorPresentation, ChatErrorPresenter
from .input import UserInput
from .models import ChatInvocation, ChatResponse, ConversationKey
from .progress import (
    ConversationPresenterFactory,
    ConversationProgressSession,
    DirectResponseTraceSink,
    NoopProgressSession,
)
from .use_case import ChatUseCase

logger = logging.getLogger(__name__)

Reply = Callable[[str], Awaitable[object]]


class ConversationResponder:
    """Run one conversation request and own its visible response lifecycle.

    The command and ambient-message adapters differ only in how they build a
    request and reply to it. Keeping the lifecycle here prevents cancellation,
    error fallback, direct-delivery tracing, and presenter cleanup from drifting
    apart between those entry points.
    """

    def __init__(
        self,
        service: ChatUseCase,
        presenters: ConversationPresenterFactory,
        errors: ChatErrorPresenter | None = None,
    ) -> None:
        self._service = service
        self._presenters = presenters
        self._errors = errors or ChatErrorPresenter()

    async def run(
        self,
        *,
        presenter_context: Any,
        key: ConversationKey,
        prompt: str,
        user_input: UserInput | None,
        invocation: ChatInvocation,
        request_ref: str,
        reply: Reply,
    ) -> bool:
        """Run a request, returning whether the model turn completed."""
        presentation = await self._open(presenter_context, request_ref)
        conversation_ref = _conversation_ref(key)
        try:
            response = await self._service.ask(
                key,
                prompt,
                user_input=user_input,
                invocation=invocation,
                progress=presentation,
            )
        except asyncio.CancelledError:
            await self.show_cancelled(presentation, reply, request_ref=request_ref)
            raise
        except Exception as exc:
            error = self._errors.present(exc, request_ref=request_ref)
            self._log_error(error, exc, conversation_ref=conversation_ref)
            await self._deliver_failure(presentation, error.message, reply, request_ref)
            return False
        else:
            await self._deliver_success(presentation, response, reply, request_ref)
            return True
        finally:
            await asyncio.shield(
                self._safe_presentation(
                    "close",
                    presentation.aclose(),
                    request_ref=request_ref,
                )
            )

    async def show_cancelled(
        self,
        presentation: ConversationProgressSession,
        reply: Reply,
        *,
        request_ref: str,
    ) -> None:
        """Report cancellation while preserving the direct-delivery trace."""
        if presentation.owns_message:
            delivered = await asyncio.shield(
                self._safe_presentation(
                    "cancel",
                    presentation.cancel(),
                    request_ref=request_ref,
                )
            )
            if delivered:
                return
        sent = await self._safe_reply(
            "cancel", reply("已取消当前文字回复。"), request_ref=request_ref
        )
        if sent is not None:
            await self._record_direct_delivery(presentation, sent, cancelled=True)

    async def reply_error(
        self,
        error: Exception,
        *,
        request_ref: str,
        reply: Reply,
        conversation_ref: str,
    ) -> None:
        """Map and send an error for short command operations."""
        presentation = self._errors.present(error, request_ref=request_ref)
        self._log_error(presentation, error, conversation_ref=conversation_ref)
        await self._safe_reply("error", reply(presentation.message), request_ref=request_ref)

    async def _deliver_failure(
        self,
        presentation: ConversationProgressSession,
        message: str,
        reply: Reply,
        request_ref: str,
    ) -> None:
        if presentation.owns_message:
            delivered = await self._safe_presentation(
                "fail",
                presentation.fail(message),
                request_ref=request_ref,
            )
            if delivered:
                return
        sent = await self._safe_reply("error", reply(message), request_ref=request_ref)
        if sent is not None:
            await self._record_direct_delivery(presentation, sent, failure_message=message)

    async def _deliver_success(
        self,
        presentation: ConversationProgressSession,
        response: ChatResponse,
        reply: Reply,
        request_ref: str,
    ) -> None:
        if presentation.owns_message:
            delivered = await self._safe_presentation(
                "complete",
                presentation.complete(response),
                request_ref=request_ref,
            )
            if delivered:
                return
        sent = await self._safe_reply("final", reply(response.content), request_ref=request_ref)
        if sent is not None:
            await self._record_direct_delivery(presentation, sent, response=response)

    async def _open(
        self,
        context: Any,
        request_ref: str,
    ) -> ConversationProgressSession:
        try:
            return await self._presenters.open(context)
        except Exception as exc:
            logger.warning(
                "Conversation presentation degraded: request_ref=%s phase=presentation "
                "responsibility=transport recoverability=fallback "
                "code=presentation_open_failed error=%s",
                request_ref,
                type(exc).__name__,
                exc_info=True,
            )
            return NoopProgressSession()

    @staticmethod
    def _log_error(
        presentation: ChatErrorPresentation,
        error: Exception,
        *,
        conversation_ref: str,
    ) -> None:
        log = logger.error if presentation.internal else logger.warning
        log(
            "Chat request failed: conversation=%s code=%s responsibility=%s reference=%s error=%s",
            conversation_ref,
            presentation.code,
            presentation.responsibility,
            presentation.reference or "none",
            type(error).__name__,
            exc_info=presentation.internal,
        )

    @staticmethod
    async def _safe_presentation(
        operation: str,
        work: Awaitable[None],
        *,
        request_ref: str,
    ) -> bool:
        try:
            await work
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Conversation presentation degraded: request_ref=%s phase=presentation "
                "responsibility=transport recoverability=fallback "
                "code=presentation_%s_failed error=%s",
                request_ref,
                operation,
                type(exc).__name__,
                exc_info=True,
            )
            return False
        return True

    @staticmethod
    async def _safe_reply(
        operation: str,
        work: Awaitable[object],
        *,
        request_ref: str,
    ) -> object | None:
        try:
            return await work
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Conversation reply delivery degraded: request_ref=%s phase=presentation "
                "responsibility=transport recoverability=discarded "
                "code=reply_%s_failed error=%s",
                request_ref,
                operation,
                type(exc).__name__,
                exc_info=True,
            )
            return None

    @staticmethod
    async def _record_direct_delivery(
        presentation: ConversationProgressSession,
        message: object,
        *,
        response: ChatResponse | None = None,
        failure_message: str = "",
        cancelled: bool = False,
    ) -> None:
        if not isinstance(presentation, DirectResponseTraceSink):
            return
        try:
            await presentation.record_delivery(
                message,
                response=response,
                failure_message=failure_message,
                cancelled=cancelled,
            )
        except Exception as exc:
            logger.warning("Direct Agent response tracking degraded: %s", type(exc).__name__)


def _conversation_ref(key: ConversationKey) -> str:
    return opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id)
