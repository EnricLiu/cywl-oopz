"""Framework-neutral values used by the AI Agent feature."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from cywl_oopz.features.chat.models import ConversationKey


class ProviderProtocol(StrEnum):
    """Provider protocols supported by the model registry."""

    OPENAI_CHAT_COMPATIBLE = "openai_chat_compatible"


class ModelCapability(StrEnum):
    """Capabilities verified for one concrete provider/model pair."""

    TOOL_CALLING = "tool_calling"
    STREAMING = "streaming"
    STRUCTURED_OUTPUT = "structured_output"
    PARALLEL_TOOLS = "parallel_tools"
    IMAGE_INPUT = "image_input"


@dataclass(frozen=True, slots=True)
class LlmProvider:
    """One provider endpoint and its application-owned credentials."""

    id: UUID
    alias: str
    display_name: str
    protocol: ProviderProtocol
    base_url: str
    api_key: str
    user_selectable: bool
    enabled: bool
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alias = self.alias.strip()
        display_name = self.display_name.strip()
        base_url = self.base_url.rstrip("/")
        if not alias or not display_name:
            raise ValueError("Provider alias and display name must not be empty")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Provider base URL must be HTTP(S)")
        if self.enabled and not self.api_key.strip():
            raise ValueError("Enabled provider API key must not be empty")
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(frozen=True, slots=True)
class LlmModel:
    """One remotely addressed model and its measured capabilities."""

    id: UUID
    provider_id: UUID
    alias: str
    remote_model_name: str
    display_name: str
    enabled: bool
    is_provider_default: bool
    is_application_default: bool
    capabilities: frozenset[ModelCapability] = field(default_factory=frozenset)
    limits: Mapping[str, Any] = field(default_factory=dict)
    fallback_model_id: UUID | None = None
    pricing: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alias = self.alias.strip()
        remote_name = self.remote_model_name.strip()
        display_name = self.display_name.strip()
        if not alias or not remote_name or not display_name:
            raise ValueError("Model alias, remote name, and display name must not be empty")
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "remote_model_name", remote_name)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "limits", MappingProxyType(dict(self.limits)))
        object.__setattr__(self, "pricing", MappingProxyType(dict(self.pricing)))


@dataclass(frozen=True, slots=True)
class AgentModelRef:
    """Run-pinned provider/model identity without credentials."""

    provider_id: UUID
    model_id: UUID
    provider_alias: str
    model_alias: str
    remote_model_name: str
    protocol: ProviderProtocol
    capabilities: frozenset[ModelCapability]
    fallback_model_id: UUID | None


class ModelSelectionSource(StrEnum):
    """Precedence source that selected a model for a run."""

    THREAD = "thread"
    USER = "user"
    CHANNEL = "channel"
    APPLICATION = "application"


@dataclass(frozen=True, slots=True)
class ModelSelectionCandidates:
    """Stable model IDs loaded from each configured selection layer."""

    thread_model_id: UUID | None = None
    user_model_id: UUID | None = None
    channel_model_id: UUID | None = None
    application_model_id: UUID | None = None

    def in_precedence_order(
        self,
    ) -> tuple[tuple[ModelSelectionSource, UUID | None], ...]:
        """Return candidates in the only precedence order used by the application."""
        return (
            (ModelSelectionSource.THREAD, self.thread_model_id),
            (ModelSelectionSource.USER, self.user_model_id),
            (ModelSelectionSource.CHANNEL, self.channel_model_id),
            (ModelSelectionSource.APPLICATION, self.application_model_id),
        )


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """Resolved run-pinned model and any skipped higher-priority sources."""

    model: AgentModelRef
    source: ModelSelectionSource
    skipped_sources: tuple[ModelSelectionSource, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    """Trusted caller identity derived from OOPZ context by the integration layer."""

    person_id: str
    conversation: ConversationKey
    is_administrator: bool = False


@dataclass(frozen=True, slots=True)
class AgentRunLimits:
    """Hard budgets for one bounded Agent run."""

    timeout_seconds: float = 45.0
    max_model_requests: int = 6
    max_tool_calls: int = 8
    max_total_tokens: int = 32_000
    max_parallel_tools: int = 3

    def __post_init__(self) -> None:
        values = (
            self.timeout_seconds,
            self.max_model_requests,
            self.max_tool_calls,
            self.max_total_tokens,
            self.max_parallel_tools,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Agent run limits must be positive")


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """Provider-neutral versioned message envelope."""

    role: str
    kind: str
    content: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.role.strip() or not self.kind.strip():
            raise ValueError("Agent message role and kind must not be empty")
        object.__setattr__(self, "content", MappingProxyType(dict(self.content)))


@dataclass(frozen=True, slots=True)
class AgentRunRequest:
    """Complete project-owned input to an Agent engine."""

    run_id: UUID
    thread_id: UUID
    identity: AgentIdentity
    model: AgentModelRef
    prompt: str
    context: tuple[AgentMessage, ...]
    enabled_tools: tuple[str, ...]
    limits: AgentRunLimits

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("Agent prompt must not be empty")


class AgentRunStatus(StrEnum):
    """Persisted lifecycle states for one Agent run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


class AgentStopReason(StrEnum):
    """Framework-neutral reason that stopped an Agent loop."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    MODEL_REQUEST_LIMIT = "model_request_limit"
    TOOL_CALL_LIMIT = "tool_call_limit"
    TOKEN_LIMIT = "token_limit"
    TOOL_DENIED = "tool_denied"
    PROVIDER_ERROR = "provider_error"
    TOOL_ERROR = "tool_error"
    INVALID_OUTPUT = "invalid_output"
    STALE_RUN_ABANDONED = "stale_run_abandoned"


_STOP_STATUS = {
    AgentStopReason.COMPLETED: AgentRunStatus.SUCCEEDED,
    AgentStopReason.CANCELLED: AgentRunStatus.CANCELLED,
    AgentStopReason.STALE_RUN_ABANDONED: AgentRunStatus.ABANDONED,
}


@dataclass(frozen=True, slots=True)
class AgentRunState:
    """Small immutable state machine used before persistence is involved."""

    run_id: UUID
    status: AgentRunStatus = AgentRunStatus.PENDING
    stop_reason: AgentStopReason | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def start(self, now: datetime) -> AgentRunState:
        """Move a newly created run into its only active state."""
        if self.status is not AgentRunStatus.PENDING:
            raise ValueError(f"Cannot start Agent run from {self.status}")
        return replace(self, status=AgentRunStatus.RUNNING, started_at=now)

    def finish(self, reason: AgentStopReason, now: datetime) -> AgentRunState:
        """Finish a running run using the canonical reason-to-status mapping."""
        if self.status is not AgentRunStatus.RUNNING:
            raise ValueError(f"Cannot finish Agent run from {self.status}")
        status = _STOP_STATUS.get(reason, AgentRunStatus.FAILED)
        return replace(self, status=status, stop_reason=reason, finished_at=now)


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Normalized result returned by an Agent engine."""

    output: str
    stop_reason: AgentStopReason
    input_tokens: int = 0
    output_tokens: int = 0
    model_requests: int = 0
    tool_calls: int = 0


@dataclass(frozen=True, slots=True)
class AgentThread:
    """Durable Agent thread metadata without loading its messages."""

    id: UUID
    key: ConversationKey
    selected_model_id: UUID | None
    expires_at: datetime
    summary: str = ""
    summary_through_sequence: int = 0
    summary_version: int = 0
    version: int = 1

    def is_expired(self, now: datetime) -> bool:
        """Return whether this thread must be reset before another run."""
        return self.expires_at <= now


@dataclass(frozen=True, slots=True)
class AgentRun:
    """Persistable run metadata pinned before model I/O starts."""

    id: UUID
    thread_id: UUID
    provider_id: UUID
    model_id: UUID
    selection_source: ModelSelectionSource
    limits: AgentRunLimits
    state: AgentRunState
    usage: Mapping[str, Any] = field(default_factory=dict)
    error_code: str = ""
    heartbeat_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))
