"""Provider-neutral values for durable delegated Agent tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class DelegatedResultStyle(StrEnum):
    BRIEF = "brief"
    DETAILED = "detailed"


class DelegatedTaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_RETRY = "waiting_retry"
    CANCEL_REQUESTED = "cancel_requested"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        return self in {
            self.SUCCEEDED,
            self.FAILED,
            self.CANCELLED,
            self.INTERRUPTED,
        }


class DelegatedTaskLane(StrEnum):
    READ_PARALLEL = "read_parallel"
    MUTATION_SERIAL = "mutation_serial"


class DelegatedTaskNotificationState(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    PRESENTED = "presented"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class DelegatedTaskPolicy:
    profile: str
    agent_model_id: UUID


@dataclass(frozen=True, slots=True)
class DelegatedTaskSubmission:
    owner_person_id: str
    area_id: str
    text_channel_id: str
    voice_channel_id: str
    origin_voice_session_id: UUID
    provider_call_id: str
    objective: str
    result_style: DelegatedResultStyle
    lane: DelegatedTaskLane
    conflict_key: str
    agent_model_id: UUID
    allowed_tool_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskRef:
    """An owner-scoped UUID or a session-scoped spoken alias such as T2."""

    task_id: UUID | None = None
    origin_voice_session_id: UUID | None = None
    session_sequence: int | None = None

    @classmethod
    def parse(cls, value: str, *, origin_voice_session_id: UUID) -> TaskRef:
        normalized = value.strip()
        alias = re.fullmatch(r"[tT](\d{1,9})", normalized)
        if alias is not None:
            sequence = int(alias.group(1))
            if sequence <= 0:
                raise ValueError("Task alias must be positive")
            return cls(
                origin_voice_session_id=origin_voice_session_id,
                session_sequence=sequence,
            )
        try:
            return cls(task_id=UUID(normalized))
        except ValueError as exc:
            raise ValueError("Task must be an alias such as T1 or a UUID") from exc


@dataclass(frozen=True, slots=True)
class TaskListQuery:
    status: DelegatedTaskStatus | None = None
    limit: int = 5
    origin_voice_session_id: UUID | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.limit <= 20:
            raise ValueError("Task list limit must be between 1 and 20")


@dataclass(frozen=True, slots=True)
class DelegatedAgentTask:
    id: UUID
    owner_person_id: str
    area_id: str
    text_channel_id: str
    voice_channel_id: str
    origin_voice_session_id: UUID
    session_sequence: int
    provider_call_id: str
    objective: str
    result_style: DelegatedResultStyle
    status: DelegatedTaskStatus
    lane: DelegatedTaskLane
    conflict_key: str
    notification_state: DelegatedTaskNotificationState
    agent_model_id: UUID
    allowed_tool_names: tuple[str, ...]
    progress_stage: str = ""
    progress_summary: str = ""
    result_summary: str = ""
    result_text: str = ""
    error_code: str = ""
    error_message: str = ""
    retry_count: int = 0
    cancel_requested_at: datetime | None = None
    next_attempt_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    agent_thread_id: UUID | None = None
    agent_run_id: UUID | None = None

    @property
    def alias(self) -> str:
        return f"T{self.session_sequence}"


@dataclass(frozen=True, slots=True)
class CancelOutcome:
    task: DelegatedAgentTask | None
    cancel_requested: bool
    already_terminal: bool = False


@dataclass(frozen=True, slots=True)
class RecoverySummary:
    requeued: int = 0
    cancelled: int = 0
    interrupted: int = 0
    task_ids: tuple[UUID, ...] = field(default_factory=tuple)
