"""Framework-neutral request helpers and safe errors for Agent commands."""

from __future__ import annotations

import logging

from cywl_oopz.commands.models import CommandRequest, CommandScope
from cywl_oopz.core.errors import (
    AuthorizationError,
    DatabaseError,
    FeatureDisabledError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitExceeded,
)
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.chat.models import ChatInvocation, ConversationKey

logger = logging.getLogger(__name__)


class AgentCommandContext:
    """Project request projection shared by Agent management handlers."""

    @staticmethod
    def conversation(request: CommandRequest) -> ConversationKey:
        private = request.location.scope is CommandScope.PRIVATE
        return ConversationKey(
            "private" if private else "channel",
            "" if private else request.location.area_id,
            "" if private else request.location.channel_id,
            request.actor.person_id,
        )

    @staticmethod
    def invocation(request: CommandRequest) -> ChatInvocation:
        recipients = tuple(
            dict.fromkeys(
                mention.person_id
                for mention in request.mentions
                if not mention.is_bot and mention.person_id != request.actor.person_id
            )
        )
        return ChatInvocation(
            request.source.message_id,
            request.location.channel_id,
            recipients,
        )

    @classmethod
    def reference(cls, request: CommandRequest) -> str:
        key = cls.conversation(request)
        return opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id)


class AgentCommandErrorPresenter:
    """Render expected Agent/chat infrastructure errors for management commands."""

    def message(self, error: Exception) -> str | None:
        if isinstance(error, FeatureDisabledError):
            return "文字对话功能当前未启用。"
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
        return None

    async def reply(self, request: CommandRequest, error: Exception) -> None:
        message = self.message(error)
        if message is None:
            raise error
        logger.warning(
            "Agent command failed: conversation=%s error=%s",
            AgentCommandContext.reference(request),
            type(error).__name__,
        )
        await request.responder.reply(message)
