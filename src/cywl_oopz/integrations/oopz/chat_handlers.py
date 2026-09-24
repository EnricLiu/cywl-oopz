"""OOPZ event handlers for conversational messages that are not commands."""

from __future__ import annotations

from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from cywl_oopz.conversation.input import UserInput as AgentUserInput
from cywl_oopz.conversation.models import ChatInvocation, ChatInvocationFactory, ConversationKey
from cywl_oopz.conversation.progress import ConversationPresenterFactory, NoopPresenterFactory
from cywl_oopz.conversation.responder import ConversationResponder
from cywl_oopz.conversation.use_case import ChatUseCase
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.storage.channel_settings import ChannelSettingsRepository

from .chat_invocation import conversation_key_from_context, invocation_from_context
from .conversation_input import OopzConversationInputFactory


class OopzChatHandlerController:
    """Share safe Agent presentation for OOPZ mention and ambient events."""

    def __init__(
        self,
        service: ChatUseCase,
        presenter_factory: ConversationPresenterFactory | None = None,
        invocation_factory: ChatInvocationFactory | None = None,
    ) -> None:
        self._service = service
        self._invocations = invocation_factory
        self._responder = ConversationResponder(
            service, presenter_factory or NoopPresenterFactory()
        )

    @staticmethod
    def _key(context: EventContext) -> ConversationKey:
        return conversation_key_from_context(context)

    def _invocation(self, context: EventContext) -> ChatInvocation:
        if self._invocations is not None:
            return self._invocations.from_context(context)
        return invocation_from_context(context)

    @staticmethod
    def _event_request_ref(context: EventContext, key: ConversationKey) -> str:
        source_message_id = str(getattr(getattr(context.event, "message", None), "message_id", ""))
        return opaque_ref(
            "chat-event",
            source_message_id,
            key.scope,
            key.area_id,
            key.channel_id,
            key.person_id,
        )

    async def _ask_with_presenter(self, context: EventContext, user_input: AgentUserInput) -> bool:
        key = self._key(context)
        return await self._responder.run(
            presenter_context=context,
            key=key,
            prompt=user_input.prompt,
            user_input=user_input,
            invocation=self._invocation(context),
            request_ref=self._event_request_ref(context, key),
            reply=context.reply,
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
        input_factory: OopzConversationInputFactory | None = None,
    ) -> None:
        super().__init__(service, presenter_factory, invocation_factory)
        self._bot_person_id = bot_person_id
        self._prefix = command_prefix
        self._input_factory = input_factory or OopzConversationInputFactory()

    async def handle(self, message: OopzMessage, context: EventContext) -> bool:
        if not self.matches(message):
            return False
        try:
            user_input = self._input_factory.from_message(message)
        except ValueError:
            await context.reply(
                f"你好！请在提及我后附上想问的内容，或使用 {self._prefix}chat <内容>。"
            )
            return True
        await self._ask_with_presenter(context, user_input)
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
        input_factory: OopzConversationInputFactory | None = None,
    ) -> None:
        super().__init__(service, presenter_factory, invocation_factory)
        self._channels = channels
        self._input_factory = input_factory or OopzConversationInputFactory()

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
        try:
            user_input = self._input_factory.from_message(message)
        except ValueError:
            return False
        await self._ask_with_presenter(context, user_input)
        return True
