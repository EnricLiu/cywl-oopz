"""Mailbox delivery workers independent of realtime conversation state."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from uuid import UUID

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.settings import VoiceSettings

from .models import (
    VoiceTaskNotification,
)
from .notifications import (
    VoiceTaskNotificationStrategy,
    compile_internal_task_context,
)
from .ports import (
    RealtimeVoiceSession,
    VoiceSessionRuntimeContext,
    VoiceTaskMailbox,
)
from .runtime_events import _MailboxAvailable, _MailboxClaimed, _MailboxPresented

logger = logging.getLogger(__name__)

MailboxEvent = _MailboxAvailable | _MailboxClaimed | _MailboxPresented
_NOTIFICATION_BATCH_LIMIT = 3
_NOTIFICATION_PERSIST_ATTEMPTS = 3


class VoiceNotificationWorker:
    """Perform mailbox I/O and return events; never mutate conversation state."""

    def __init__(
        self,
        context: VoiceSessionRuntimeContext,
        settings: VoiceSettings,
        mailbox: VoiceTaskMailbox | None,
        publish: Callable[[MailboxEvent], Awaitable[None]],
        *,
        coalesce_seconds: float,
        persist_retry_seconds: float,
    ) -> None:
        self._context = context
        self._settings = settings
        self._task_mailbox = mailbox
        self._publish = publish
        self._coalesce_seconds = coalesce_seconds
        self._persist_retry_seconds = persist_retry_seconds

    async def listen(self) -> None:
        mailbox = self._task_mailbox
        if mailbox is None:
            return
        await self._publish(_MailboxAvailable())
        while True:
            signalled = await mailbox.wait(
                self._context.descriptor.owner_person_id,
                self._settings.mailbox_poll_seconds,
            )
            if signalled:
                await asyncio.sleep(self._coalesce_seconds)
            await self._publish(_MailboxAvailable())

    async def claim(self) -> None:
        mailbox = self._task_mailbox
        if mailbox is None:
            return
        notices: tuple[VoiceTaskNotification, ...] = ()
        error_kind = ""
        try:
            notices = await mailbox.claim(
                self._context.descriptor.session_id,
                _NOTIFICATION_BATCH_LIMIT,
            )
            await self._publish(_MailboxClaimed(notices))
        except asyncio.CancelledError:
            if notices:
                await asyncio.shield(mailbox.defer(tuple(item.task_id for item in notices)))
            raise
        except Exception as exc:
            error_kind = exception_kind(exc)
            logger.warning(
                "Voice task mailbox claim failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                error_kind,
                exc_info=True,
            )
            await self._publish(_MailboxClaimed((), error_kind))

    async def request_proactive(
        self,
        notices: tuple[VoiceTaskNotification, ...],
        session: RealtimeVoiceSession | None,
    ) -> None:
        if session is None:
            await self._proactive_request_failed(notices, "provider_session_missing")
            return
        try:
            await session.request_proactive_response(compile_internal_task_context(notices))
        except asyncio.CancelledError:
            await asyncio.shield(
                self._require_mailbox().defer(tuple(item.task_id for item in notices))
            )
            raise
        except Exception as exc:
            logger.warning(
                "Voice proactive task notification request failed: session=%s tasks=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                len(notices),
                exception_kind(exc),
                exc_info=True,
            )
            await self._proactive_request_failed(notices, exception_kind(exc))

    async def _proactive_request_failed(
        self,
        notices: tuple[VoiceTaskNotification, ...],
        error_kind: str,
    ) -> None:
        logger.warning(
            "Voice proactive task notification request failed: session=%s tasks=%s error=%s",
            opaque_ref(str(self._context.descriptor.session_id)),
            len(notices),
            error_kind,
        )
        await self.defer(tuple(item.task_id for item in notices))
        await self._publish(
            _MailboxPresented(
                notices,
                VoiceTaskNotificationStrategy.INTERNAL_RESPONSE,
                False,
            )
        )

    async def mark_presented(
        self,
        notices: tuple[VoiceTaskNotification, ...],
    ) -> None:
        succeeded = False
        task_ids = tuple(item.task_id for item in notices)
        for attempt in range(1, _NOTIFICATION_PERSIST_ATTEMPTS + 1):
            try:
                await asyncio.shield(self._require_mailbox().mark_presented(task_ids))
                succeeded = True
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log = logger.error if attempt == _NOTIFICATION_PERSIST_ATTEMPTS else logger.warning
                log(
                    "Could not persist started proactive notification: "
                    "session=%s tasks=%s attempt=%s error=%s",
                    opaque_ref(str(self._context.descriptor.session_id)),
                    len(notices),
                    attempt,
                    exception_kind(exc),
                    exc_info=True,
                )
                if attempt < _NOTIFICATION_PERSIST_ATTEMPTS:
                    await asyncio.sleep(self._persist_retry_seconds * attempt)
        await self._publish(
            _MailboxPresented(
                notices,
                VoiceTaskNotificationStrategy.INTERNAL_RESPONSE,
                succeeded,
            )
        )

    async def defer(self, task_ids: tuple[UUID, ...]) -> None:
        mailbox = self._task_mailbox
        if mailbox is None:
            return
        try:
            await asyncio.shield(mailbox.defer(task_ids))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Voice task notification defer failed: session=%s tasks=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                len(task_ids),
                exception_kind(exc),
                exc_info=True,
            )

    async def present_text(
        self,
        notices: tuple[VoiceTaskNotification, ...],
    ) -> None:
        mailbox = self._task_mailbox
        if mailbox is None:
            return
        succeeded = False
        try:
            succeeded = await mailbox.present_text(notices)
        except asyncio.CancelledError:
            await asyncio.shield(mailbox.defer(tuple(item.task_id for item in notices)))
            raise
        except Exception as exc:
            logger.warning(
                "Voice task text fallback failed: session=%s tasks=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                len(notices),
                exception_kind(exc),
                exc_info=True,
            )
            await self.defer(tuple(item.task_id for item in notices))
        else:
            if not succeeded:
                await self.defer(tuple(item.task_id for item in notices))
        await self._publish(
            _MailboxPresented(
                notices,
                VoiceTaskNotificationStrategy.TEXT_FALLBACK,
                succeeded,
            )
        )

    def _require_mailbox(self) -> VoiceTaskMailbox:
        mailbox = self._task_mailbox
        if mailbox is None:  # pragma: no cover - guarded by notification call sites
            raise RuntimeError("Voice task mailbox is unavailable")
        return mailbox
