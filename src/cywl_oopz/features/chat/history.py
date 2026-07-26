"""Deterministic history truncation that keeps the newest useful context."""

from __future__ import annotations

from cywl_oopz.core.errors import CywlError

from .models import ChatMessage, ChatRole


class ChatInputTooLongError(CywlError):
    """Raised before a single user prompt would exceed the configured context budget."""


class HistoryTrimmer:
    """Bounds persisted history by message count and text character count."""

    def __init__(self, max_messages: int, max_characters: int) -> None:
        self._max_messages = max_messages
        self._max_characters = max_characters

    def trim(self, messages: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...]:
        """Retain the newest messages that fit entirely inside both budgets."""
        selected: list[ChatMessage] = []
        character_count = 0
        for message in reversed(messages):
            if len(selected) >= self._max_messages:
                break
            next_count = character_count + len(message.content)
            if next_count > self._max_characters:
                break
            selected.append(message)
            character_count = next_count
        selected.reverse()
        while selected and selected[0].role is ChatRole.ASSISTANT:
            selected.pop(0)
        return tuple(selected)

    def validate_input(self, content: str) -> None:
        """Avoid silently rewriting a user's current prompt to fit the context."""
        if len(content) > self._max_characters:
            raise ChatInputTooLongError("The prompt exceeds the configured history budget")
