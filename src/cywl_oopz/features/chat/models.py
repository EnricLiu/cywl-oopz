"""Value objects used by the text-chat feature."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from cywl_oopz.conversation.models import ChatInvocation as ChatInvocation
from cywl_oopz.conversation.models import ChatInvocationFactory as ChatInvocationFactory
from cywl_oopz.conversation.models import ChatResponse as ChatResponse
from cywl_oopz.conversation.models import ChatStatus as ChatStatus
from cywl_oopz.conversation.models import ConversationKey as ConversationKey
from cywl_oopz.core.errors import ProviderResponseError


class ChatRole(StrEnum):
    """Roles supported by OpenAI-compatible chat-completion APIs."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One validated message in a conversation transcript."""

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Chat message content must not be empty")

    def to_payload(self) -> dict[str, str]:
        """Return the provider-neutral JSON representation."""
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> ChatMessage:
        """Parse a persisted or provider message without accepting malformed data."""
        try:
            role = ChatRole(str(value.get("role", "")))
        except ValueError as exc:
            raise ProviderResponseError("Chat message has an unknown role") from exc
        content = value.get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError("Chat message has no text content")
        return cls(role=role, content=content)


@dataclass(frozen=True, slots=True)
class ConversationSession:
    """Persisted, expiring chat state for one conversation key."""

    key: ConversationKey
    messages: tuple[ChatMessage, ...]
    selected_model: str | None
    expires_at: datetime

    def is_expired(self, now: datetime) -> bool:
        """Return whether the history must be discarded before use."""
        return self.expires_at <= now


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A provider-neutral request containing only the necessary context."""

    model: str
    messages: tuple[ChatMessage, ...]
    user_id: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One incremental response delta from a streaming provider."""

    delta: str = ""
    model: str = ""
    finish_reason: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
