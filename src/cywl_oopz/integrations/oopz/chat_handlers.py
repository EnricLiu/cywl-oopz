"""OOPZ event handlers for conversational messages that are not commands."""

from __future__ import annotations

import asyncio
import logging

from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.chat.commands import ChatCommandController
from cywl_oopz.features.chat.models import ChatInvocation, ChatInvocationFactory, ConversationKey
from cywl_oopz.features.chat.progress import (
    ConversationPresenterFactory,
    ConversationProgressSession,
    DirectResponseTraceSink,
    NoopProgressSession,
)
from cywl_oopz.features.chat.use_case import ChatUseCase
from cywl_oopz.storage.channel_settings import ChannelSettingsRepository

logger = logging.getLogger(__name__)


class OopzChatHandlerController(ChatCommandController):
    """Share safe Agent presentation for OOPZ mention and ambient events."""

    @staticmethod
    def _key(context: EventContext) -> ConversationKey:
        return ConversationKey.from_oopz_context(context)

    def _invocation(self, context: EventContext) -> ChatInvocation:
        if self._invocations is not None:
            return self._invocations.from_context(context)
        return ChatInvocation.from_oopz_context(context)

    async def _ask_with_presenter(self, context: EventContext, prompt: str) -> bool:
        try:
            presentation = await self._presenters.open(context)
        except Exception as exc:
            logger.warning("Conversation presenter failed to open: %s", type(exc).__name__)
            presentation = NoopProgressSession()
        key = self._key(context)
        try:
            response = await self._service.ask(
                key,
                prompt,
                invocation=self._invocation(context),
                progress=presentation,
            )
        except asyncio.CancelledError:
            await self._show_cancelled(context, presentation)
            raise
        except Exception as exc:
            conversation_ref = opaque_ref(
                key.scope,
                key.area_id,
                key.channel_id,
                key.person_id,
            )
            source_message_id = str(
                getattr(getattr(context.event, "message", None), "message_id", "")
            )
            error_presentation = self._error_presentation(
                exc,
                request_ref=opaque_ref(
                    "chat-event",
                    source_message_id,
                    key.scope,
                    key.area_id,
                    key.channel_id,
                    key.person_id,
                ),
            )
            self._log_error(error_presentation, exc, conversation_ref=conversation_ref)
            message = error_presentation.message
            if presentation.owns_message:
                await presentation.fail(message)
            else:
                sent = await context.reply(message)
                await self._record_direct_delivery(
                    presentation,
                    sent,
                    failure_message=message,
                )
            return False
        else:
            if presentation.owns_message:
                await presentation.complete(response)
            else:
                sent = await context.reply(response.content)
                await self._record_direct_delivery(presentation, sent, response=response)
            return True
        finally:
            await asyncio.shield(presentation.aclose())

    @staticmethod
    async def _show_cancelled(
        context: EventContext,
        presentation: ConversationProgressSession,
    ) -> None:
        if presentation.owns_message:
            await asyncio.shield(presentation.cancel())
            return
        sent = await context.reply("已取消当前文字回复。")
        if isinstance(presentation, DirectResponseTraceSink):
            try:
                await presentation.record_delivery(sent, cancelled=True)
            except Exception as exc:
                logger.warning(
                    "Cancelled Agent response tracking degraded: %s",
                    type(exc).__name__,
                )


class MentionChatHandler(OopzChatHandlerController):
    """Reply only when an incoming non-command message explicitly mentions this bot."""

    def __init__(
        self,
        service: ChatUseCase,
        bot_person_id: str,
        presenter_factory: ConversationPresenterFactory | None = None,
        invocation_factory: ChatInvocationFactory | None = None,
        *,
        command_prefix: str = "/",
    ) -> None:
        super().__init__(service, presenter_factory, invocation_factory)
        self._bot_person_id = bot_person_id
        self._prefix = command_prefix

    async def handle(self, message: OopzMessage, context: EventContext) -> bool:
        if not self.matches(message):
            return False
        prompt = (message.plain_text or message.text or message.content).strip()
        if not prompt:
            await context.reply(
                f"你好！请在提及我后附上想问的内容，或使用 {self._prefix}chat <内容>。"
            )
            return True
        await self._ask_with_presenter(context, prompt)
        return True

    def matches(self, message: OopzMessage) -> bool:
        mentions = getattr(message, "mention_list", ())
        return any(
            str(getattr(mention, "person", "")) == self._bot_person_id for mention in mentions
        )


class AmbientChatHandler(OopzChatHandlerController):
    """Handle private messages and channels explicitly enabled in PostgreSQL."""

    def __init__(
        self,
        service: ChatUseCase,
        channels: ChannelSettingsRepository,
        presenter_factory: ConversationPresenterFactory | None = None,
        invocation_factory: ChatInvocationFactory | None = None,
    ) -> None:
        super().__init__(service, presenter_factory, invocation_factory)
        self._channels = channels

    async def matches(self, message: OopzMessage, context: EventContext) -> bool:
        if not self._service.enabled:
            return False
        if bool(getattr(context.event, "is_private", False)):
            return True
        area_id = str(getattr(message, "area", "")).strip()
        channel_id = str(getattr(message, "channel", "")).strip()
        if not area_id or not channel_id:
            return False
        return await self._channels.is_chat_enabled(area_id, channel_id)

    async def handle(self, message: OopzMessage, context: EventContext) -> bool:
        prompt = (message.plain_text or message.text or message.content).strip()
        if not prompt:
            return False
        await self._ask_with_presenter(context, prompt)
        return True
