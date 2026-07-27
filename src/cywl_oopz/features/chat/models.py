"""Value objects used by the text-chat feature."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

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
class ConversationKey:
    """Stable, privacy-preserving scope for one person's conversation."""

    scope: str
    area_id: str
    channel_id: str
    person_id: str

    @classmethod
    def from_oopz_context(cls, context: Any) -> ConversationKey:
        """Build a key from the SDK event context required by a chat command."""
        event = getattr(context, "event", None)
        message = getattr(event, "message", None)
        if message is None:
            raise ValueError("A chat command requires an OOPZ message event")

        person_id = str(getattr(message, "sender_id", "")).strip()
        if not person_id:
            raise ValueError("Message sender is required for a chat command")
        if bool(getattr(event, "is_private", False)):
            return cls(scope="private", area_id="", channel_id="", person_id=person_id)

        area_id = str(getattr(message, "area", "")).strip()
        channel_id = str(getattr(message, "channel", "")).strip()
        if not area_id or not channel_id:
            raise ValueError("Channel messages require area and channel identifiers")
        return cls(
            scope="channel",
            area_id=area_id,
            channel_id=channel_id,
            person_id=person_id,
        )


@dataclass(frozen=True, slots=True)
class ChatInvocation:
    """Provider-neutral metadata for side effects targeting the source message."""

    source_message_id: str
    transport_channel_id: str

    @classmethod
    def from_oopz_context(cls, context: Any) -> ChatInvocation:
        """Extract only stable message targeting values from the SDK boundary."""
        event = getattr(context, "event", None)
        message = getattr(event, "message", None)
        if message is None:
            raise ValueError("A chat invocation requires an OOPZ message event")
        return cls(
            source_message_id=str(getattr(message, "message_id", "")).strip(),
            transport_channel_id=str(getattr(message, "channel", "")).strip(),
        )


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
class ChatResponse:
    """A complete model response with optional provider usage metadata."""

    content: str
    model: str
    finish_reason: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None
    model_requests: int | None = None
    tool_calls: int | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ProviderResponseError("LLM response is empty")
        numeric_values = (
            self.input_tokens,
            self.output_tokens,
            self.elapsed_seconds,
            self.model_requests,
            self.tool_calls,
        )
        if any(value is not None and value < 0 for value in numeric_values):
            raise ProviderResponseError("LLM response metrics must not be negative")


@dataclass(frozen=True, slots=True)
class ChatChunk:
    """One incremental response delta from a streaming provider."""

    delta: str = ""
    model: str = ""
    finish_reason: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ChatStatus:
    """Safe status data shown by the `!chat-status` command."""

    enabled: bool
    active: bool
    model: str
    history_message_count: int
    expires_at: datetime | None
    cooldown_seconds: float
