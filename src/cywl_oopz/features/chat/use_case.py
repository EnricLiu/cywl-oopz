"""Controller-facing text conversation contract shared by legacy and Agent modes."""

from __future__ import annotations

from typing import Protocol

from cywl_oopz.features.agent.input import AgentUserInput

from .models import ChatInvocation, ChatResponse, ChatStatus, ConversationKey
from .progress import ProgressSink


class ChatUseCase(Protocol):
    """Methods required by existing OOPZ chat controllers."""

    @property
    def enabled(self) -> bool:
        """Return whether incoming chat triggers should be active."""

    async def ask(
        self,
        key: ConversationKey,
        prompt: str,
        *,
        user_input: AgentUserInput | None = None,
        invocation: ChatInvocation | None = None,
        progress: ProgressSink | None = None,
    ) -> ChatResponse:
        """Answer one prompt."""

    async def clear(self, key: ConversationKey) -> None:
        """Start a new scoped conversation."""

    async def select_model(self, key: ConversationKey, model: str) -> str:
        """Select one model for the current conversation."""

    async def status(self, key: ConversationKey) -> ChatStatus:
        """Return safe conversation metadata."""
