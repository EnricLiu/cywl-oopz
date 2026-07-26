"""Framework-light values for registered Agent tools and their executions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from cywl_oopz.features.agent.models import AgentIdentity, AgentRunLimits

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


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


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Stable tool schema and code-enforced execution policy."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    effect: ToolEffect
    version: str = "1"
    timeout_seconds: float = 10.0
    max_retries: int = 0
    max_output_characters: int = 32_768
    concurrency_safe: bool = False
    idempotent: bool = False

    def __post_init__(self) -> None:
        name = self.name.strip()
        description = self.description.strip()
        version = self.version.strip()
        if not _TOOL_NAME.fullmatch(name):
            raise ValueError("Tool name must be a stable snake_case identifier")
        if not description or not version:
            raise ValueError("Tool description and version must not be empty")
        if self.timeout_seconds <= 0 or self.max_output_characters < 128:
            raise ValueError(
                "Tool timeout must be positive and output limit at least 128 characters"
            )
        if self.max_retries < 0:
            raise ValueError("Tool max_retries must not be negative")
        if self.effect is not ToolEffect.READ and not self.idempotent:
            raise ValueError("Write and admin tools must declare idempotency")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "version", version)

    @property
    def sequential(self) -> bool:
        """Writes and non-concurrency-safe reads must execute as barriers."""
        return self.effect is not ToolEffect.READ or not self.concurrency_safe


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Trusted project context supplied to tools, never model-controlled."""

    run_id: UUID
    identity: AgentIdentity
    limits: AgentRunLimits
    enabled_tools: tuple[str, ...]
    model_requests_used: int = 0
    tool_calls_used: int = 0


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One validated-by-name model request before argument validation."""

    call_id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.name.strip():
            raise ValueError("Tool call ID and name must not be empty")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Durable execution record independent from a framework message type."""

    id: UUID
    run_id: UUID
    call_id: str
    tool_name: str
    tool_version: str
    effect: ToolEffect
    status: ToolExecutionStatus
    idempotency_key: str
    input_payload: Mapping[str, Any]
    output_payload: Mapping[str, Any] | None
    error_code: str
    started_at: datetime
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_payload",
            MappingProxyType(dict(self.input_payload)),
        )
        if self.output_payload is not None:
            object.__setattr__(
                self,
                "output_payload",
                MappingProxyType(dict(self.output_payload)),
            )


@dataclass(frozen=True, slots=True)
class ToolExecutionClaim:
    """Result of atomically claiming a run/tool-call identity."""

    execution: ToolExecution
    created: bool


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Safe, bounded result returned to the model adapter."""

    call_id: str
    tool_name: str
    status: ToolExecutionStatus
    output: Mapping[str, Any] = field(default_factory=dict)
    error_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", MappingProxyType(dict(self.output)))

    def model_payload(self) -> dict[str, object]:
        """Return the only envelope exposed to the model."""
        if self.status is ToolExecutionStatus.SUCCEEDED:
            return {"ok": True, "data": dict(self.output)}
        return {
            "ok": False,
            "error": self.error_code or self.status.value,
        }
