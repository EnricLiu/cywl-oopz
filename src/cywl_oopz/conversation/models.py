"""Transport-neutral conversation values shared by chat, Agent, and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from cywl_oopz.core.errors import ProviderResponseError


@dataclass(frozen=True, slots=True)
class ConversationKey:
    """Stable, privacy-preserving scope for one person's conversation."""

    scope: str
    area_id: str
    channel_id: str
    person_id: str


@dataclass(frozen=True, slots=True)
class ChatInvocation:
    """Provider-neutral metadata for side effects targeting the source message."""

    source_message_id: str
    transport_channel_id: str
    mentioned_person_ids: tuple[str, ...] = ()


class ChatInvocationFactory(Protocol):
    """Build trusted transport metadata from one OOPZ event context."""

    def from_context(self, context: Any) -> ChatInvocation:
        """Return one normalized invocation."""

    def from_request(self, request: Any) -> ChatInvocation:
        """Return trusted metadata from a framework-neutral command request."""


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
    image_count: int | None = None
    image_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ProviderResponseError("LLM response is empty")
        numeric_values = (
            self.input_tokens,
            self.output_tokens,
            self.elapsed_seconds,
            self.model_requests,
            self.tool_calls,
            self.image_count,
            self.image_bytes,
        )
        if any(value is not None and value < 0 for value in numeric_values):
            raise ProviderResponseError("LLM response metrics must not be negative")


@dataclass(frozen=True, slots=True)
class ChatStatus:
    """Safe status data shown by the `/chat-status` command."""

    enabled: bool
    active: bool
    model: str
    history_message_count: int
    expires_at: datetime | None
    cooldown_seconds: float


@dataclass(frozen=True, slots=True)
class ActorContext:
    """Trusted caller and message identity shared by feature entry points."""

    person_id: str
    conversation: ConversationKey
    source_message_id: str = ""
    transport_channel_id: str = ""
    mentioned_person_ids: tuple[str, ...] = field(default=(), repr=False)
