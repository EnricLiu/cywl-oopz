"""SQLAlchemy models owned by the CYWL application."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    FetchedValue,
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

from cywl_oopz.core.lifecycle import (
    AgentRunStatus,
    AgentStopReason,
    ModelSelectionSource,
    ToolEffect,
    ToolExecutionStatus,
)

CURRENT_TIMESTAMP = text("CURRENT_TIMESTAMP")
GENERATED_UUID = text("gen_random_uuid()")
EMPTY_STRING = text("''")
EMPTY_JSON = text("'[]'::json")
EMPTY_JSONB_ARRAY = text("'[]'::jsonb")
EMPTY_JSONB_OBJECT = text("'{}'::jsonb")
DEFAULT_AGENT_TOOLS = text(
    "'["
    '"get_agent_status",'
    '"get_channel_settings",'
    '"react_to_message",'
    '"search_music_catalog",'
    '"enqueue_music",'
    '"get_music_queue",'
    '"skip_music",'
    '"pause_music",'
    '"resume_music",'
    '"search_web",'
    '"read_web_page",'
    '"set_music_playback_mode",'
    '"create_music_playlist",'
    '"list_music_playlists",'
    '"get_music_playlist",'
    '"add_music_playlist_track",'
    '"remove_music_playlist_track",'
    '"load_music_playlist"'
    "]'::jsonb"
)
TRUE = text("true")
FALSE = text("false")


def _enum_values(enum_type: type[StrEnum]) -> list[str]:
    return [member.value for member in enum_type]


AGENT_RUN_STATUS_ENUM = Enum(
    AgentRunStatus,
    name="agent_run_status",
    values_callable=_enum_values,
    validate_strings=True,
)
AGENT_STOP_REASON_ENUM = Enum(
    AgentStopReason,
    name="agent_stop_reason",
    values_callable=_enum_values,
    validate_strings=True,
)
MODEL_SELECTION_SOURCE_ENUM = Enum(
    ModelSelectionSource,
    name="model_selection_source",
    values_callable=_enum_values,
    validate_strings=True,
)
TOOL_EFFECT_ENUM = Enum(
    ToolEffect,
    name="tool_effect",
    values_callable=_enum_values,
    validate_strings=True,
)
TOOL_EXECUTION_STATUS_ENUM = Enum(
    ToolExecutionStatus,
    name="tool_execution_status",
    values_callable=_enum_values,
    validate_strings=True,
)


def utc_now() -> datetime:
    """Return an aware UTC timestamp for application-managed audit fields."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all application tables."""


class ChannelSettingsRecord(Base):
    """Durable feature settings for one OOPZ area/channel pair."""

    __tablename__ = "channel_settings"
    __table_args__ = (UniqueConstraint("area_id", "channel_id"),)

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    area_id: Mapped[str] = mapped_column(String(128), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    chat_enabled: Mapped[bool] = mapped_column(
        default=False,
        server_default=FALSE,
        nullable=False,
    )
    enabled_agent_tools: Mapped[list[str]] = mapped_column(
        JSONB,
        default=lambda: [
            "get_agent_status",
            "get_channel_settings",
            "react_to_message",
            "search_music_catalog",
            "enqueue_music",
            "get_music_queue",
            "skip_music",
            "pause_music",
            "resume_music",
            "search_web",
            "read_web_page",
            "set_music_playback_mode",
            "create_music_playlist",
            "list_music_playlists",
            "get_music_playlist",
            "add_music_playlist_track",
            "remove_music_playlist_track",
            "load_music_playlist",
        ],
        server_default=DEFAULT_AGENT_TOOLS,
        nullable=False,
    )
    default_model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("llm_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
    )


class MusicPlaylistRecord(Base):
    """One shared playlist identified inside an OOPZ area."""

    __tablename__ = "music_playlists"
    __table_args__ = (
        UniqueConstraint("area_id", "normalized_name"),
        Index("ix_music_playlists_area_updated", "area_id", "updated_at"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    area_id: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(80), nullable=False)
    created_by_person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
    )


class MusicPlaylistTrackRecord(Base):
    """One ordered catalog metadata snapshot inside a shared playlist."""

    __tablename__ = "music_playlist_tracks"
    __table_args__ = (
        UniqueConstraint(
            "playlist_id",
            "position",
            deferrable=True,
            initially="DEFERRED",
        ),
        CheckConstraint("position > 0", name="ck_music_playlist_tracks_position_positive"),
        CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_music_playlist_tracks_duration_nonnegative",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    playlist_id: Mapped[UUID] = mapped_column(
        ForeignKey("music_playlists.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    artists: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        server_default=EMPTY_JSONB_ARRAY,
        nullable=False,
    )
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    added_by_person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
    )


class ConversationSessionRecord(Base):
    """Short-lived LLM conversation history scoped to an OOPZ user context."""

    __tablename__ = "conversation_sessions"
    __table_args__ = (UniqueConstraint("scope", "area_id", "channel_id", "person_id"),)

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    area_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="",
        server_default=EMPTY_STRING,
    )
    channel_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="",
        server_default=EMPTY_STRING,
    )
    person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    messages: Mapped[list[dict[str, str]]] = mapped_column(
        JSON,
        default=list,
        server_default=EMPTY_JSON,
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
    )


class RateLimitBucketRecord(Base):
    """Reserved durable rate-limit state for future multi-process deployment."""

    __tablename__ = "rate_limit_buckets"

    bucket_key: Mapped[str] = mapped_column(String(256), primary_key=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
    )
    hit_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(
        Text,
        default="",
        server_default=EMPTY_STRING,
        nullable=False,
    )


class LlmProviderRecord(Base):
    """Provider endpoint and credentials maintained by the bot owner."""

    __tablename__ = "llm_providers"

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    alias: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    protocol: Mapped[str] = mapped_column(String(64), nullable=False)
    base_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    api_key: Mapped[str] = mapped_column(Text, nullable=False)
    user_selectable: Mapped[bool] = mapped_column(
        default=True,
        server_default=TRUE,
        nullable=False,
    )
    enabled: Mapped[bool] = mapped_column(
        default=True,
        server_default=TRUE,
        nullable=False,
    )
    config: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        default=dict,
        server_default=EMPTY_JSONB_OBJECT,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
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

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    provider_id: Mapped[UUID] = mapped_column(
        ForeignKey("llm_providers.id", ondelete="CASCADE"),
        nullable=False,
    )
    alias: Mapped[str] = mapped_column(String(128), nullable=False)
    remote_model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    enabled: Mapped[bool] = mapped_column(
        default=True,
        server_default=TRUE,
        nullable=False,
    )
    is_provider_default: Mapped[bool] = mapped_column(
        default=False,
        server_default=FALSE,
        nullable=False,
    )
    is_application_default: Mapped[bool] = mapped_column(
        default=False,
        server_default=FALSE,
        nullable=False,
    )
    capabilities: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        server_default=EMPTY_JSONB_ARRAY,
        nullable=False,
    )
    limits: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        default=dict,
        server_default=EMPTY_JSONB_OBJECT,
        nullable=False,
    )
    fallback_model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("llm_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    pricing: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        default=dict,
        server_default=EMPTY_JSONB_OBJECT,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
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
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
    )


class AgentThreadRecord(Base):
    """Expiring provider-neutral Agent conversation state."""

    __tablename__ = "agent_threads"
    __table_args__ = (UniqueConstraint("scope", "area_id", "channel_id", "person_id"),)

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    scope: Mapped[str] = mapped_column(String(32), nullable=False)
    area_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="",
        server_default=EMPTY_STRING,
    )
    channel_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="",
        server_default=EMPTY_STRING,
    )
    person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    selected_model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("llm_models.id", ondelete="SET NULL"),
        nullable=True,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    summary: Mapped[str] = mapped_column(
        Text,
        default="",
        server_default=EMPTY_STRING,
        nullable=False,
    )
    summary_through_sequence: Mapped[int] = mapped_column(
        BigInteger,
        default=0,
        server_default=text("0"),
        nullable=False,
    )
    summary_version: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
    )
    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        server_default=text("1"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
    )


class AgentMemoryPreferenceRecord(Base):
    """One user's explicit long-term-memory on/off choice."""

    __tablename__ = "agent_memory_preferences"

    person_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    enabled: Mapped[bool] = mapped_column(
        default=True,
        server_default=TRUE,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
    )


class AgentMemoryItemRecord(Base):
    """One expiring long-term memory item owned by exactly one OOPZ person."""

    __tablename__ = "agent_memory_items"
    __table_args__ = (
        Index(
            "ix_agent_memory_items_owner_updated",
            "owner_person_id",
            "updated_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    owner_person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    namespace: Mapped[str] = mapped_column(
        String(64),
        default="explicit",
        server_default=text("'explicit'"),
        nullable=False,
    )
    content: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    source_thread_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("agent_threads.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_message_sequence: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class AgentRunRecord(Base):
    """One pinned and bounded Agent execution."""

    __tablename__ = "agent_runs"

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    thread_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_threads.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[AgentRunStatus] = mapped_column(
        AGENT_RUN_STATUS_ENUM,
        default=AgentRunStatus.PENDING,
        server_default=text("'pending'"),
        nullable=False,
    )
    stop_reason: Mapped[AgentStopReason | None] = mapped_column(
        AGENT_STOP_REASON_ENUM,
        nullable=True,
    )
    provider_id: Mapped[UUID] = mapped_column(
        ForeignKey("llm_providers.id", ondelete="RESTRICT"),
        nullable=False,
    )
    model_id: Mapped[UUID] = mapped_column(
        ForeignKey("llm_models.id", ondelete="RESTRICT"),
        nullable=False,
    )
    selection_source: Mapped[ModelSelectionSource] = mapped_column(
        MODEL_SELECTION_SOURCE_ENUM,
        nullable=False,
    )
    limits: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        default=dict,
        server_default=EMPTY_JSONB_OBJECT,
        nullable=False,
    )
    usage: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        default=dict,
        server_default=EMPTY_JSONB_OBJECT,
        nullable=False,
    )
    error_code: Mapped[str] = mapped_column(
        String(128),
        default="",
        server_default=EMPTY_STRING,
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        nullable=False,
    )


class AgentMessageRecord(Base):
    """One ordered provider-neutral message in an Agent thread."""

    __tablename__ = "agent_messages"
    __table_args__ = (UniqueConstraint("thread_id", "sequence"),)

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
    )


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

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    run_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    tool_call_id: Mapped[str] = mapped_column(String(256), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(64), nullable=False)
    effect: Mapped[ToolEffect] = mapped_column(TOOL_EFFECT_ENUM, nullable=False)
    status: Mapped[ToolExecutionStatus] = mapped_column(
        TOOL_EXECUTION_STATUS_ENUM,
        default=ToolExecutionStatus.STARTED,
        server_default=text("'started'"),
        nullable=False,
    )
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    input_payload: Mapped[dict[str, object]] = mapped_column("input", JSONB, nullable=False)
    output_payload: Mapped[dict[str, object] | None] = mapped_column(
        "output",
        JSONB,
        nullable=True,
    )
    error_code: Mapped[str] = mapped_column(
        String(128),
        default="",
        server_default=EMPTY_STRING,
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
