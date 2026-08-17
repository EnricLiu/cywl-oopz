"""Framework-neutral request helpers and safe errors for Agent commands."""

from __future__ import annotations

import logging

from cywl_oopz.commands.models import CommandRequest, CommandScope
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.chat.error_presenter import ChatErrorPresenter
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

    def __init__(self) -> None:
        self._presenter = ChatErrorPresenter()

    def message(self, error: Exception, *, request_ref: str = "agent-command") -> str:
        return self._presenter.present(error, request_ref=request_ref).message

    async def reply(self, request: CommandRequest, error: Exception) -> None:
        request_ref = opaque_ref(
            "agent-command",
            request.source.message_id,
            AgentCommandContext.reference(request),
        )
        presentation = self._presenter.present(error, request_ref=request_ref)
        log = logger.error if presentation.internal else logger.warning
        log(
            "Agent command failed: conversation=%s code=%s responsibility=%s reference=%s error=%s",
            AgentCommandContext.reference(request),
            presentation.code,
            presentation.responsibility,
            presentation.reference or "none",
            type(error).__name__,
            exc_info=presentation.internal,
        )
        await request.responder.reply(presentation.message)
