"""Stable lifecycle enums shared by the Agent domain and persistence layer."""

from enum import StrEnum


class ModelSelectionSource(StrEnum):
    """Precedence source that selected a model for a run."""

    THREAD = "thread"
    USER = "user"
    CHANNEL = "channel"
    APPLICATION = "application"


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


class ToolEffect(StrEnum):
    """Observable side-effect class used by deterministic policy."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ToolExecutionStatus(StrEnum):
    """Persisted lifecycle of one model-issued tool call."""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"
