"""Composition root for the bot application."""

from __future__ import annotations

import logging
from collections.abc import Coroutine
from typing import Any

from oopz_sdk import OopzBot
from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from .commands.builtin import HelpCommand, PingCommand, StatusCommand
from .commands.router import CommandRouter
from .core.errors import DatabaseError
from .core.health import HealthRegistry, HealthState
from .features.chat.commands import (
    AmbientChatHandler,
    CancelChatCommand,
    ChatCommand,
    ChatStatusCommand,
    MentionChatHandler,
    ModelCommand,
    NewConversationCommand,
)
from .features.chat.models import ConversationKey
from .features.chat.openai_compatible import OpenAICompatibleChatProvider
from .features.chat.provider import ChatProvider, DisabledChatProvider
from .features.chat.repository import SqlAlchemyConversationRepository
from .features.chat.service import ChatService
from .features.chat.tasks import ChatTaskSupervisor
from .settings import AppSettings
from .storage.channel_settings import SqlAlchemyChannelSettingsRepository
from .storage.database import Database

logger = logging.getLogger(__name__)


class BotApplication:
    """Owns OOPZ integration, application resources, and feature services."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.health = HealthRegistry()
        self.database = Database(settings.database)
        self.bot = OopzBot(settings.oopz)
        self.commands = CommandRouter(settings.command_prefix)
        self._provider = self._create_chat_provider()
        self.chat_tasks = ChatTaskSupervisor()
        self.chat = ChatService(
            settings.chat,
            self._provider,
            SqlAlchemyConversationRepository(self.database.session_factory),
            health=self.health,
        )
        channel_settings = SqlAlchemyChannelSettingsRepository(self.database.session_factory)
        self._mention_handler = MentionChatHandler(self.chat, settings.oopz.person_uid)
        self._ambient_handler = AmbientChatHandler(self.chat, channel_settings)
        self._register_commands()
        self.bot.on_ready(self._on_ready)
        self.bot.on_message(self._on_message)
        self.health.mark("database", HealthState.PENDING)
        self.health.mark(
            "llm",
            HealthState.PENDING if settings.chat.enabled else HealthState.DISABLED,
        )
        self.health.mark("oopz", HealthState.PENDING)

    def _create_chat_provider(self) -> ChatProvider:
        if not self.settings.chat.enabled:
            return DisabledChatProvider()
        return OpenAICompatibleChatProvider(self.settings.chat)

    def _register_commands(self) -> None:
        self.commands.register(PingCommand())
        self.commands.register(HelpCommand(self.commands))
        self.commands.register(StatusCommand(self.health))
        self.commands.register(ChatCommand(self.chat))
        self.commands.register(NewConversationCommand(self.chat, self.chat_tasks))
        self.commands.register(CancelChatCommand(self.chat, self.chat_tasks))
        self.commands.register(ModelCommand(self.chat, self.chat_tasks))
        self.commands.register(ChatStatusCommand(self.chat))

    async def run(self) -> None:
        """Start the database check before entering the long-running OOPZ client."""
        try:
            try:
                await self.database.start()
            except DatabaseError:
                self.health.mark("database", HealthState.DEGRADED, "connection check failed")
                raise
            self.health.mark("database", HealthState.HEALTHY, "connection check passed")
            await self.bot.run()
        finally:
            await self.chat_tasks.close()
            await self._provider.aclose()
            await self.database.close()

    async def _on_ready(self, _: EventContext) -> None:
        self.health.mark("oopz", HealthState.HEALTHY, "websocket connected")
        logger.info("Bot connected; command prefix is %r", self.settings.command_prefix)

    async def _on_message(self, message: OopzMessage, context: EventContext) -> None:
        """Route short commands inline and own slow LLM work in supervised tasks."""
        logger.debug(f"[OnMessage] Bot: new message: {message}")
        command = self.commands.parse(message.plain_text or message.text or message.content)
        if command is not None:
            if command.name == "chat":
                await self._start_chat_task(context, self.commands.dispatch(message, context))
                return
            await self.commands.dispatch(message, context)
            return
        if self._mention_handler.matches(message):
            await self._start_chat_task(context, self._mention_handler.handle(message, context))
            return
        try:
            ambient_enabled = await self._ambient_handler.matches(message, context)
        except DatabaseError:
            logger.exception("Failed to evaluate ambient chat policy")
            return
        if ambient_enabled:
            await self._start_chat_task(context, self._ambient_handler.handle(message, context))

    async def _start_chat_task(
        self,
        context: EventContext,
        operation: Coroutine[Any, Any, object],
    ) -> None:
        """Register slow work so the SDK receive loop can immediately process later events."""
        try:
            key = ConversationKey.from_oopz_context(context)
        except ValueError:
            operation.close()
            await context.reply("无法识别当前对话的位置，请稍后重试。")
            return
        if not self.chat_tasks.start(key, operation):
            await context.reply("当前对话正在生成回复；可使用 !cancel 取消后再试。")
