"""Chat-specific naming for the application task supervisor."""

from __future__ import annotations

from cywl_oopz.core.tasks import TaskSupervisor

from .models import ConversationKey


class ChatTaskSupervisor(TaskSupervisor[ConversationKey]):
    """Own at most one LLM reply task per conversation."""

    def __init__(self) -> None:
        super().__init__(lambda key: f"chat:{key.scope}")
