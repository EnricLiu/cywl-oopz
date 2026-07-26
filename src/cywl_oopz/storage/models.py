"""SQLAlchemy models owned by the CYWL application."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import Uuid


def utc_now() -> datetime:
    """Return an aware UTC timestamp for application-managed audit fields."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all application tables."""


class ChannelSettingsRecord(Base):
    """Durable feature settings for one OOPZ area/channel pair."""

    __tablename__ = "channel_settings"
    __table_args__ = (UniqueConstraint("area_id", "channel_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    area_id: Mapped[str] = mapped_column(String(128), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    chat_enabled: Mapped[bool] = mapped_column(default=False, nullable=False)
    enabled_agent_tools: Mapped[list[str]] = mapped_column(
        JSONB,
        default=lambda: [
            "get_agent_status",
            "get_channel_settings",
            "react_to_message",
        ],
        nullable=False,
    )
    default_model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("llm_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class ConversationSessionRecord(Base):
    """Short-lived LLM conversation history scoped to an OOPZ user context."""

    __tablename__ = "conversation_sessions"
    __table_args__ = (UniqueConstraint("scope", "area_id", "channel_id", "person_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    area_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    channel_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    messages: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class RateLimitBucketRecord(Base):
    """Reserved durable rate-limit state for future multi-process deployment."""

    __tablename__ = "rate_limit_buckets"

    bucket_key: Mapped[str] = mapped_column(String(256), primary_key=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)


class LlmProviderRecord(Base):
    """Provider endpoint and credentials maintained by the bot owner."""

    __tablename__ = "llm_providers"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    alias: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    protocol: Mapped[str] = mapped_column(String(64), nullable=False)
    base_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    api_key: Mapped[str] = mapped_column(Text, nullable=False)
    user_selectable: Mapped[bool] = mapped_column(default=True, nullable=False)
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    config: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class LlmModelRecord(Base):
    """One model available through an LLM provider."""

    __tablename__ = "llm_models"
    __table_args__ = (
        UniqueConstraint("provider_id", "alias"),
        Index(
            "uq_llm_models_provider_default",
            "provider_id",
            unique=True,
            postgresql_where=text("is_provider_default"),
        ),
        Index(
            "uq_llm_models_application_default",
            "is_application_default",
            unique=True,
            postgresql_where=text("is_application_default"),
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    provider_id: Mapped[UUID] = mapped_column(
        ForeignKey("llm_providers.id", ondelete="CASCADE"),
        nullable=False,
    )
    alias: Mapped[str] = mapped_column(String(128), nullable=False)
    remote_model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_provider_default: Mapped[bool] = mapped_column(default=False, nullable=False)
    is_application_default: Mapped[bool] = mapped_column(default=False, nullable=False)
    capabilities: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    limits: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    fallback_model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("llm_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    pricing: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class UserLlmPreferenceRecord(Base):
    """One user's default model for newly resolved Agent threads."""

    __tablename__ = "user_llm_preferences"

    person_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    preferred_model_id: Mapped[UUID] = mapped_column(
        ForeignKey("llm_models.id", ondelete="CASCADE"),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class AgentThreadRecord(Base):
    """Expiring provider-neutral Agent conversation state."""

    __tablename__ = "agent_threads"
    __table_args__ = (UniqueConstraint("scope", "area_id", "channel_id", "person_id"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    area_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    channel_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("llm_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    summary_through_sequence: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    summary_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )


class AgentRunRecord(Base):
    """One pinned and bounded Agent execution."""

    __tablename__ = "agent_runs"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    thread_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_threads.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_id: Mapped[UUID] = mapped_column(
        ForeignKey("llm_providers.id", ondelete="RESTRICT"),
        nullable=False,
    )
    model_id: Mapped[UUID] = mapped_column(
        ForeignKey("llm_models.id", ondelete="RESTRICT"),
        nullable=False,
    )
    selection_source: Mapped[str] = mapped_column(String(32), nullable=False)
    limits: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    usage: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    error_code: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AgentMessageRecord(Base):
    """One ordered provider-neutral message in an Agent thread."""

    __tablename__ = "agent_messages"
    __table_args__ = (UniqueConstraint("thread_id", "sequence"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    thread_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_threads.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AgentToolExecutionRecord(Base):
    """Idempotent record for one model-requested tool call."""

    __tablename__ = "agent_tool_executions"
    __table_args__ = (
        UniqueConstraint("run_id", "tool_call_id"),
        Index(
            "uq_agent_tool_executions_idempotency",
            "run_id",
            "idempotency_key",
            unique=True,
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    tool_call_id: Mapped[str] = mapped_column(String(256), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(64), nullable=False)
    effect: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    input_payload: Mapped[dict[str, object]] = mapped_column("input", JSONB, nullable=False)
    output_payload: Mapped[dict[str, object] | None] = mapped_column(
        "output",
        JSONB,
        nullable=True,
    )
    error_code: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
