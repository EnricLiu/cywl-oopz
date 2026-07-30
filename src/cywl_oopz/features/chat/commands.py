"""OOPZ command controllers for the text-chat use cases."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC

from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message as OopzMessage

from cywl_oopz.commands.router import ParsedCommand
from cywl_oopz.core.errors import (
    AuthorizationError,
    DatabaseError,
    FeatureDisabledError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitExceeded,
)
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.storage.channel_settings import ChannelSettingsRepository

from .history import ChatInputTooLongError
from .models import ChatInvocation, ChatInvocationFactory, ConversationKey
from .progress import (
    ConversationPresenterFactory,
    ConversationProgressSession,
    NoopPresenterFactory,
    NoopProgressSession,
)
from .tasks import ChatTaskSupervisor
from .use_case import ChatUseCase

logger = logging.getLogger(__name__)


class ChatCommandController:
    """Shared safe error mapping for chat-facing command controllers."""

    def __init__(
        self,
        service: ChatUseCase,
        presenter_factory: ConversationPresenterFactory | None = None,
        invocation_factory: ChatInvocationFactory | None = None,
    ) -> None:
        self._service = service
        self._presenters = presenter_factory or NoopPresenterFactory()
        self._invocations = invocation_factory

    @staticmethod
    def _key(context: EventContext) -> ConversationKey:
        return ConversationKey.from_oopz_context(context)

    @staticmethod
    def _error_message(error: Exception) -> str:
        if isinstance(error, FeatureDisabledError):
            return "文字对话功能当前未启用。"
        if isinstance(error, ChatInputTooLongError):
            return "这条消息太长，请缩短后再试。"
        if isinstance(error, RateLimitExceeded):
            if error.retry_after_seconds > 0:
                return f"请求过于频繁，请在 {error.retry_after_seconds:.1f} 秒后重试。"
            return "当前对话请求较多，请稍后重试。"
        if isinstance(error, ProviderTimeoutError):
            return "模型响应超时，请稍后重试。"
        if isinstance(error, ProviderError):
            return "模型服务暂时不可用，请稍后重试。"
        if isinstance(error, DatabaseError):
            return "会话服务暂时不可用，请稍后重试。"
        if isinstance(error, AuthorizationError):
            return "你没有执行此操作的权限。"
        if isinstance(error, ValueError):
            return "命令参数不正确，请使用 !help 查看用法。"
        logger.error("Unexpected chat command failure: error=%s", type(error).__name__)
        return "处理请求时出现了问题，请稍后重试。"

    def _invocation(self, context: EventContext) -> ChatInvocation:
        if self._invocations is not None:
            return self._invocations.from_context(context)
        return ChatInvocation.from_oopz_context(context)

    async def _reply_error(self, context: EventContext, error: Exception) -> None:
        logger.warning(
            "Chat command failed: conversation=%s error=%s",
            self._conversation_ref(context),
            type(error).__name__,
        )
        await context.reply(self._error_message(error))

    async def _ask_with_presenter(
        self,
        context: EventContext,
        prompt: str,
    ) -> bool:
        """Run one shared chat path and keep terminal output in the owned message."""
        try:
            presentation = await self._presenters.open(context)
        except Exception as exc:
            logger.warning(
                "Conversation presenter failed to open: %s",
                type(exc).__name__,
            )
            presentation = NoopProgressSession()
        try:
            response = await self._service.ask(
                self._key(context),
                prompt,
                invocation=self._invocation(context),
                progress=presentation,
            )
        except asyncio.CancelledError:
            await self._show_cancelled(context, presentation)
            raise
        except Exception as exc:
            logger.warning(
                "Chat request failed: conversation=%s error=%s",
                self._conversation_ref(context),
                type(exc).__name__,
            )
            message = self._error_message(exc)
            if presentation.owns_message:
                await presentation.fail(message)
            else:
                await context.reply(message)
            return False
        else:
            if presentation.owns_message:
                await presentation.complete(response)
            else:
                await context.reply(response.content)
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
        else:
            await context.reply("已取消当前文字回复。")

    @staticmethod
    def _conversation_ref(context: EventContext) -> str:
        key = ConversationKey.from_oopz_context(context)
        return opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id)


class ChatCommand(ChatCommandController):
    """Start or continue a text conversation with `!chat <prompt>`."""

    name = "chat"
    description = "向 LLM 发起或继续文字对话。"

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        prompt = " ".join(command.arguments)
        if not prompt.strip():
            await context.reply("用法：!chat <想说的话>")
            return
        await self._ask_with_presenter(context, prompt)


class MentionChatHandler(ChatCommandController):
    """Reply only when an incoming non-command message explicitly mentions this bot."""

    def __init__(
        self,
        service: ChatUseCase,
        bot_person_id: str,
        presenter_factory: ConversationPresenterFactory | None = None,
        invocation_factory: ChatInvocationFactory | None = None,
    ) -> None:
        super().__init__(service, presenter_factory, invocation_factory)
        self._bot_person_id = bot_person_id

    async def handle(self, message: OopzMessage, context: EventContext) -> bool:
        """Handle a bot mention, returning whether the message was consumed."""
        if not self.matches(message):
            return False
        prompt = (message.plain_text or message.text or message.content).strip()
        if not prompt:
            await context.reply("你好！请在提及我后附上想问的内容，或使用 !chat <内容>。")
            return True
        await self._ask_with_presenter(context, prompt)
        return True

    def matches(self, message: OopzMessage) -> bool:
        """Return whether this incoming message explicitly mentions the bot."""
        mentions = getattr(message, "mention_list", ())
        return any(
            str(getattr(mention, "person", "")) == self._bot_person_id for mention in mentions
        )


class AmbientChatHandler(ChatCommandController):
    """Handle normal private messages and channels explicitly enabled in PostgreSQL."""

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
        """Apply the persisted trigger policy without inspecting message content."""
        if not self._service.enabled:
            return False
        event = getattr(context, "event", None)
        if bool(getattr(event, "is_private", False)):
            return True
        area_id = str(getattr(message, "area", "")).strip()
        channel_id = str(getattr(message, "channel", "")).strip()
        if not area_id or not channel_id:
            return False
        return await self._channels.is_chat_enabled(area_id, channel_id)

    async def handle(self, message: OopzMessage, context: EventContext) -> bool:
        """Answer a message already accepted by ``matches``."""
        prompt = (message.plain_text or message.text or message.content).strip()
        if not prompt:
            return False
        await self._ask_with_presenter(context, prompt)
        return True


class NewConversationCommand(ChatCommandController):
    """Forget only the caller's active conversation with `!new`."""

    name = "new"
    description = "清空当前文字对话的上下文。"

    def __init__(
        self,
        service: ChatUseCase,
        tasks: ChatTaskSupervisor,
    ) -> None:
        super().__init__(service)
        self._tasks = tasks

    async def execute(self, _: ParsedCommand, context: EventContext) -> None:
        try:
            key = self._key(context)
            await self._tasks.cancel(key)
            await self._service.clear(key)
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        await context.reply("已开始新的对话。")


class CancelChatCommand(ChatCommandController):
    """Cancel the caller's active LLM response with `!cancel`."""

    name = "cancel"
    description = "取消当前正在生成的文字回复。"

    def __init__(
        self,
        service: ChatUseCase,
        tasks: ChatTaskSupervisor,
        *,
        active_message_reports_cancel: bool = False,
    ) -> None:
        super().__init__(service)
        self._tasks = tasks
        self._active_message_reports_cancel = active_message_reports_cancel

    async def execute(self, _: ParsedCommand, context: EventContext) -> None:
        try:
            cancelled = await self._tasks.cancel(self._key(context))
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        if cancelled:
            if not self._active_message_reports_cancel:
                await context.reply("已取消当前文字回复。")
        else:
            await context.reply("当前没有正在生成的文字回复。")


class ModelCommand(ChatCommandController):
    """Show or change an allow-listed model with `!model [name]`."""

    name = "model"
    description = "查看或切换允许使用的模型。"

    def __init__(self, service: ChatUseCase, tasks: ChatTaskSupervisor) -> None:
        super().__init__(service)
        self._tasks = tasks

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        try:
            key = self._key(context)
            if not command.arguments:
                status = await self._service.status(key)
                if not status.enabled:
                    await context.reply("文字对话功能当前未启用。")
                    return
                await context.reply(f"当前模型：{status.model}")
                return
            if len(command.arguments) != 1:
                await context.reply("用法：!model [模型名称]")
                return
            if self._tasks.has_active(key):
                await context.reply("当前正在生成回复；请等待完成或先使用 !cancel。")
                return
            selected = await self._service.select_model(key, command.arguments[0])
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        await context.reply(f"当前模型已切换为：{selected}")


class ChatStatusCommand(ChatCommandController):
    """Show safe conversation metadata with `!chat-status`."""

    name = "chat-status"
    description = "查看文字对话状态，不显示聊天内容。"

    async def execute(self, _: ParsedCommand, context: EventContext) -> None:
        try:
            status = await self._service.status(self._key(context))
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        if not status.enabled:
            await context.reply("文字对话功能当前未启用。")
            return

        lines = [
            "文字对话状态：已启用",
            f"当前模型：{status.model}",
            f"会话：{'进行中' if status.active else '尚未开始'}",
            f"已保留消息数：{status.history_message_count}",
        ]
        if status.expires_at is not None:
            expires_at = status.expires_at.astimezone(UTC).isoformat(timespec="seconds")
            lines.append(f"会话过期时间（UTC）：{expires_at}")
        if status.cooldown_seconds > 0:
            lines.append(f"冷却剩余：{status.cooldown_seconds:.1f} 秒")
        await context.reply("\n".join(lines))
