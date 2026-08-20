"""SQLAlchemy models owned by the CYWL application."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    FetchedValue,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import Uuid

from cywl_oopz.core.lifecycle import (
    AgentRunStatus,
    AgentStopReason,
    ModelSelectionSource,
    ToolEffect,
    ToolExecutionStatus,
)
from cywl_oopz.features.access.models import AccessRole, RoleBindingScope
from cywl_oopz.features.admin.models import (
    OopzMessageScope,
    OutboundMessageKind,
    OutboundMessageState,
)
from cywl_oopz.features.agent.delegation.models import (
    DelegatedResultStyle,
    DelegatedTaskLane,
    DelegatedTaskNotificationState,
    DelegatedTaskStatus,
)
from cywl_oopz.features.agent.skills.models import (
    SkillOwnershipKind,
    SkillResourceKind,
    SkillShareStatus,
)
from cywl_oopz.features.voice.settings import (
    PersistedVoiceSessionStatus,
    VoiceDuplexMode,
    VoiceModelMode,
    VoiceProviderProtocol,
    VoiceTurnRole,
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
    '"clear_music_queue",'
    '"search_web",'
    '"read_web_page",'
    '"set_music_playback_mode",'
    '"create_music_playlist",'
    '"list_music_playlists",'
    '"get_music_playlist",'
    '"add_music_playlist_track",'
    '"remove_music_playlist_track",'
    '"rename_music_playlist",'
    '"delete_music_playlist",'
    '"clear_music_playlist",'
    '"load_music_playlist",'
    '"load_agent_skill",'
    '"read_agent_skill_resource",'
    '"list_agent_skill_library",'
    '"inspect_agent_skill",'
    '"create_agent_skill",'
    '"update_agent_skill",'
    '"manage_agent_skill_resource",'
    '"set_agent_skill_state",'
    '"invite_agent_skill_share",'
    '"respond_agent_skill_share",'
    '"revoke_agent_skill_share",'
    '"preview_netease_playlist",'
    '"import_netease_playlist"'
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
AGENT_SKILL_RESOURCE_KIND_ENUM = Enum(
    SkillResourceKind,
    name="agent_skill_resource_kind",
    values_callable=_enum_values,
    validate_strings=True,
)
AGENT_SKILL_OWNERSHIP_KIND_ENUM = Enum(
    SkillOwnershipKind,
    name="agent_skill_ownership_kind",
    values_callable=_enum_values,
    validate_strings=True,
)
AGENT_SKILL_SHARE_STATUS_ENUM = Enum(
    SkillShareStatus,
    name="agent_skill_share_status",
    values_callable=_enum_values,
    validate_strings=True,
)
VOICE_PROVIDER_PROTOCOL_ENUM = Enum(
    VoiceProviderProtocol,
    name="voice_provider_protocol",
    values_callable=_enum_values,
    validate_strings=True,
)
VOICE_MODEL_MODE_ENUM = Enum(
    VoiceModelMode,
    name="voice_model_mode",
    values_callable=_enum_values,
    validate_strings=True,
)
VOICE_DUPLEX_MODE_ENUM = Enum(
    VoiceDuplexMode,
    name="voice_duplex_mode",
    values_callable=_enum_values,
    validate_strings=True,
)
VOICE_SESSION_STATUS_ENUM = Enum(
    PersistedVoiceSessionStatus,
    name="voice_session_status",
    values_callable=_enum_values,
    validate_strings=True,
)
VOICE_TURN_ROLE_ENUM = Enum(
    VoiceTurnRole,
    name="voice_turn_role",
    values_callable=_enum_values,
    validate_strings=True,
)
DELEGATED_RESULT_STYLE_ENUM = Enum(
    DelegatedResultStyle,
    name="delegated_result_style",
    values_callable=_enum_values,
    validate_strings=True,
)
DELEGATED_TASK_STATUS_ENUM = Enum(
    DelegatedTaskStatus,
    name="delegated_task_status",
    values_callable=_enum_values,
    validate_strings=True,
)
DELEGATED_TASK_LANE_ENUM = Enum(
    DelegatedTaskLane,
    name="delegated_task_lane",
    values_callable=_enum_values,
    validate_strings=True,
)
TASK_NOTIFICATION_STATE_ENUM = Enum(
    DelegatedTaskNotificationState,
    name="task_notification_state",
    values_callable=_enum_values,
    validate_strings=True,
)
RBAC_ROLE_ENUM = Enum(
    AccessRole,
    name="rbac_role",
    values_callable=_enum_values,
    validate_strings=True,
)
RBAC_SCOPE_ENUM = Enum(
    RoleBindingScope,
    name="rbac_scope",
    values_callable=_enum_values,
    validate_strings=True,
)
OOPZ_MESSAGE_SCOPE_ENUM = Enum(
    OopzMessageScope,
    name="oopz_message_scope",
    values_callable=_enum_values,
    validate_strings=True,
)
OOPZ_OUTBOUND_MESSAGE_KIND_ENUM = Enum(
    OutboundMessageKind,
    name="oopz_outbound_message_kind",
    values_callable=_enum_values,
    validate_strings=True,
)
OOPZ_OUTBOUND_MESSAGE_STATE_ENUM = Enum(
    OutboundMessageState,
    name="oopz_outbound_message_state",
    values_callable=_enum_values,
    validate_strings=True,
)


def utc_now() -> datetime:
    """Return an aware UTC timestamp for application-managed audit fields."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all application tables."""


class RbacRoleBindingRecord(Base):
    """One scoped role assignment for an OOPZ message sender."""

    __tablename__ = "rbac_role_bindings"
    __table_args__ = (
        UniqueConstraint(
            "subject_person_id",
            "role",
            "scope",
            "area_id",
            "channel_id",
            name="uq_rbac_role_bindings_assignment",
        ),
        CheckConstraint(
            "(scope = 'global' AND area_id = '' AND channel_id = '') OR "
            "(scope = 'area' AND area_id <> '' AND channel_id = '') OR "
            "(scope = 'channel' AND area_id <> '' AND channel_id <> '')",
            name="ck_rbac_role_bindings_scope_address",
        ),
        CheckConstraint(
            "role <> 'owner' OR scope = 'global'",
            name="ck_rbac_role_bindings_owner_global",
        ),
        Index(
            "ix_rbac_role_bindings_subject_scope",
            "subject_person_id",
            "scope",
            "area_id",
            "channel_id",
        ),
        Index(
            "ix_rbac_role_bindings_resource",
            "scope",
            "area_id",
            "channel_id",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    subject_person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[AccessRole] = mapped_column(RBAC_ROLE_ENUM, nullable=False)
    scope: Mapped[RoleBindingScope] = mapped_column(RBAC_SCOPE_ENUM, nullable=False)
    area_id: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    channel_id: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    granted_by_person_id: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )


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
            "clear_music_queue",
            "search_web",
            "read_web_page",
            "set_music_playback_mode",
            "create_music_playlist",
            "list_music_playlists",
            "get_music_playlist",
            "add_music_playlist_track",
            "remove_music_playlist_track",
            "rename_music_playlist",
            "delete_music_playlist",
            "clear_music_playlist",
            "load_music_playlist",
            "load_agent_skill",
            "read_agent_skill_resource",
            "list_agent_skill_library",
            "inspect_agent_skill",
            "create_agent_skill",
            "update_agent_skill",
            "manage_agent_skill_resource",
            "set_agent_skill_state",
            "invite_agent_skill_share",
            "respond_agent_skill_share",
            "revoke_agent_skill_share",
            "preview_netease_playlist",
            "import_netease_playlist",
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


class AgentSkillRecord(Base):
    """One builtin or user-owned progressive-disclosure Agent skill."""

    __tablename__ = "agent_skills"
    __table_args__ = (
        Index(
            "ux_agent_skills_builtin_name",
            "name",
            unique=True,
            postgresql_where=text("ownership_kind = 'builtin'"),
        ),
        Index(
            "ux_agent_skills_personal_owner_name",
            "owner_person_id",
            "name",
            unique=True,
            postgresql_where=text("ownership_kind = 'personal'"),
        ),
        CheckConstraint(
            "name ~ '^[a-z][a-z0-9-]{0,63}$'",
            name="ck_agent_skills_name",
        ),
        CheckConstraint(
            "char_length(btrim(display_name)) > 0",
            name="ck_agent_skills_display_name",
        ),
        CheckConstraint(
            "char_length(btrim(description)) > 0 AND char_length(description) <= 1024",
            name="ck_agent_skills_description",
        ),
        CheckConstraint(
            "char_length(btrim(instructions)) > 0 AND char_length(instructions) <= 20000",
            name="ck_agent_skills_instructions",
        ),
        CheckConstraint(
            "char_length(btrim(version)) > 0",
            name="ck_agent_skills_version",
        ),
        CheckConstraint("revision > 0", name="ck_agent_skills_revision_positive"),
        CheckConstraint(
            "cywl_valid_agent_skill_required_tools(required_tools)",
            name="ck_agent_skills_required_tools",
        ),
        CheckConstraint(
            "jsonb_typeof(metadata) = 'object'",
            name="ck_agent_skills_metadata_object",
        ),
        CheckConstraint(
            "(ownership_kind = 'builtin' AND owner_person_id IS NULL) OR "
            "(ownership_kind = 'personal' AND "
            "char_length(btrim(owner_person_id)) > 0)",
            name="ck_agent_skills_ownership",
        ),
        CheckConstraint(
            "ownership_kind = 'builtin' OR enabled = (archived_at IS NULL)",
            name="ck_agent_skills_personal_state",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(String(1024), nullable=False)
    instructions: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    revision: Mapped[int] = mapped_column(
        BigInteger,
        default=1,
        server_default=text("1"),
        nullable=False,
    )
    required_tools: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        server_default=EMPTY_JSONB_ARRAY,
        nullable=False,
    )
    skill_metadata: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        server_default=EMPTY_JSONB_OBJECT,
        nullable=False,
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=TRUE,
        nullable=False,
    )
    ownership_kind: Mapped[SkillOwnershipKind] = mapped_column(
        AGENT_SKILL_OWNERSHIP_KIND_ENUM,
        default=SkillOwnershipKind.BUILTIN,
        server_default=text("'builtin'"),
        nullable=False,
    )
    owner_person_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )


class AgentSkillResourceRecord(Base):
    """One non-executable text resource belonging to an Agent skill."""

    __tablename__ = "agent_skill_resources"
    __table_args__ = (
        UniqueConstraint("skill_id", "key"),
        UniqueConstraint("skill_id", "position"),
        CheckConstraint(
            "key ~ '^[a-z][a-z0-9-]{0,159}$'",
            name="ck_agent_skill_resources_key",
        ),
        CheckConstraint(
            "char_length(btrim(display_name)) > 0",
            name="ck_agent_skill_resources_display_name",
        ),
        CheckConstraint(
            "char_length(btrim(description)) > 0",
            name="ck_agent_skill_resources_description",
        ),
        CheckConstraint(
            "char_length(btrim(content)) > 0 AND char_length(content) <= 20000",
            name="ck_agent_skill_resources_content",
        ),
        CheckConstraint(
            "media_type IN ('text/markdown', 'text/plain', 'application/json')",
            name="ck_agent_skill_resources_media_type",
        ),
        CheckConstraint(
            "position > 0",
            name="ck_agent_skill_resources_position_positive",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    skill_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_skills.id", ondelete="CASCADE"),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(String(160), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    kind: Mapped[SkillResourceKind] = mapped_column(
        AGENT_SKILL_RESOURCE_KIND_ENUM,
        nullable=False,
    )
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )


class AgentSkillShareRecord(Base):
    """One recipient invitation or accepted read grant for a personal Skill."""

    __tablename__ = "agent_skill_shares"
    __table_args__ = (
        UniqueConstraint("skill_id", "recipient_person_id"),
        Index(
            "ix_agent_skill_shares_recipient_status",
            "recipient_person_id",
            "status",
        ),
        Index("ix_agent_skill_shares_skill_status", "skill_id", "status"),
        CheckConstraint(
            "(status = 'pending' AND responded_at IS NULL) OR "
            "(status IN ('accepted', 'declined') AND responded_at IS NOT NULL)",
            name="ck_agent_skill_shares_response",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    skill_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_skills.id", ondelete="CASCADE"),
        nullable=False,
    )
    recipient_person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[SkillShareStatus] = mapped_column(
        AGENT_SKILL_SHARE_STATUS_ENUM,
        default=SkillShareStatus.PENDING,
        server_default=text("'pending'"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        nullable=False,
    )
    responded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
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
            "ux_llm_models_one_provider_default",
            "provider_id",
            unique=True,
            postgresql_where=text("is_provider_default"),
        ),
        Index(
            "ux_llm_models_one_application_default",
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
        comment="Many-to-one owner; multiple models may reference the same provider.",
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
    __table_args__ = (
        CheckConstraint(
            "jsonb_typeof(diagnostics) = 'object'",
            name="ck_agent_runs_diagnostics_object",
        ),
    )

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
    diagnostics: Mapped[dict[str, object]] = mapped_column(
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


class AgentMediaAssetRecord(Base):
    """Validated binary media attached to one durable Agent message."""

    __tablename__ = "agent_media_assets"
    __table_args__ = (
        UniqueConstraint("message_id", "ordinal"),
        Index("ix_agent_media_assets_message_id", "message_id"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str] = mapped_column(String(128), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    source_file_key: Mapped[str] = mapped_column(
        String(512), default="", server_default=EMPTY_STRING
    )
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


class OopzOutboundMessageRecord(Base):
    """Address and diagnostic linkage for one message sent by this Bot."""

    __tablename__ = "oopz_outbound_messages"
    __table_args__ = (
        CheckConstraint(
            "(scope = 'channel' AND area_id <> '' AND channel_id <> '' "
            "AND target_person_id = '') OR "
            "(scope = 'private' AND area_id = '' AND channel_id <> '' "
            "AND target_person_id <> '')",
            name="ck_oopz_outbound_messages_scope_address",
        ),
        CheckConstraint(
            "jsonb_typeof(diagnostic_snapshot) = 'object'",
            name="ck_oopz_outbound_messages_diagnostic_object",
        ),
        Index(
            "ix_oopz_outbound_messages_address",
            "scope",
            "area_id",
            "channel_id",
            "target_person_id",
        ),
        Index("ix_oopz_outbound_messages_agent_run", "agent_run_id"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4,
        server_default=GENERATED_UUID,
    )
    message_id: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    message_timestamp: Mapped[str] = mapped_column(
        String(64), default="", server_default=EMPTY_STRING, nullable=False
    )
    kind: Mapped[OutboundMessageKind] = mapped_column(
        OOPZ_OUTBOUND_MESSAGE_KIND_ENUM,
        nullable=False,
    )
    state: Mapped[OutboundMessageState] = mapped_column(
        OOPZ_OUTBOUND_MESSAGE_STATE_ENUM,
        default=OutboundMessageState.FINAL,
        server_default=text("'final'"),
        nullable=False,
    )
    scope: Mapped[OopzMessageScope] = mapped_column(OOPZ_MESSAGE_SCOPE_ENUM, nullable=False)
    area_id: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    target_person_id: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    in_reply_to_message_id: Mapped[str] = mapped_column(
        String(256), default="", server_default=EMPTY_STRING, nullable=False
    )
    owner_person_id: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    agent_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True
    )
    diagnostic_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        default=dict,
        server_default=EMPTY_JSONB_OBJECT,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )
    recalled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class VoiceProviderRecord(Base):
    """Realtime voice Provider endpoint and credential bundle."""

    __tablename__ = "voice_providers"
    __table_args__ = (
        CheckConstraint(
            "jsonb_typeof(credentials) = 'object'", name="ck_voice_providers_credentials_object"
        ),
        CheckConstraint("jsonb_typeof(config) = 'object'", name="ck_voice_providers_config_object"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid4, server_default=GENERATED_UUID
    )
    alias: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    protocol: Mapped[VoiceProviderProtocol] = mapped_column(
        VOICE_PROVIDER_PROTOCOL_ENUM, nullable=False
    )
    endpoint: Mapped[str] = mapped_column(String(2048), nullable=False)
    credentials: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default=EMPTY_JSONB_OBJECT, nullable=False
    )
    config: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default=EMPTY_JSONB_OBJECT, nullable=False
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=TRUE, nullable=False
    )
    user_selectable: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=TRUE, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=CURRENT_TIMESTAMP, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )


class VoiceModelRecord(Base):
    """One realtime model available through a voice Provider."""

    __tablename__ = "voice_models"
    __table_args__ = (
        UniqueConstraint("provider_id", "alias"),
        UniqueConstraint("provider_id", "remote_model_name"),
        Index(
            "ux_voice_models_one_provider_default",
            "provider_id",
            unique=True,
            postgresql_where=text("is_provider_default"),
        ),
        Index(
            "ux_voice_models_one_application_default",
            "is_application_default",
            unique=True,
            postgresql_where=text("is_application_default"),
        ),
        CheckConstraint(
            "jsonb_typeof(capabilities) = 'object'", name="ck_voice_models_capabilities_object"
        ),
        CheckConstraint(
            "jsonb_typeof(audio_config) = 'object'", name="ck_voice_models_audio_config_object"
        ),
        CheckConstraint(
            "jsonb_typeof(prompt_config) = 'object'", name="ck_voice_models_prompt_config_object"
        ),
        CheckConstraint("jsonb_typeof(limits) = 'object'", name="ck_voice_models_limits_object"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid4, server_default=GENERATED_UUID
    )
    provider_id: Mapped[UUID] = mapped_column(
        ForeignKey("voice_providers.id", ondelete="CASCADE"), nullable=False
    )
    alias: Mapped[str] = mapped_column(String(128), nullable=False)
    remote_model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    mode: Mapped[VoiceModelMode] = mapped_column(
        VOICE_MODEL_MODE_ENUM,
        default=VoiceModelMode.NATIVE_REALTIME,
        server_default=text("'native_realtime'"),
        nullable=False,
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=TRUE, nullable=False
    )
    is_provider_default: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=FALSE, nullable=False
    )
    is_application_default: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=FALSE, nullable=False
    )
    capabilities: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default=EMPTY_JSONB_OBJECT, nullable=False
    )
    audio_config: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default=EMPTY_JSONB_OBJECT, nullable=False
    )
    prompt_config: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default=EMPTY_JSONB_OBJECT, nullable=False
    )
    limits: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default=EMPTY_JSONB_OBJECT, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=CURRENT_TIMESTAMP, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )


class VoiceUserPreferenceRecord(Base):
    """One user's model, speaker voice, and duplex defaults for new sessions."""

    __tablename__ = "voice_user_preferences"

    owner_person_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    preferred_model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("voice_models.id", ondelete="SET NULL"), nullable=True
    )
    voice_id: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    duplex_mode: Mapped[VoiceDuplexMode] = mapped_column(
        VOICE_DUPLEX_MODE_ENUM,
        default=VoiceDuplexMode.FULL,
        server_default=text("'full'"),
        nullable=False,
    )
    delegated_agent_model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("llm_models.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=CURRENT_TIMESTAMP, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )


class VoiceChannelSettingsRecord(Base):
    """Realtime voice policy for one OOPZ area voice channel."""

    __tablename__ = "voice_channel_settings"
    __table_args__ = (
        CheckConstraint(
            "idle_timeout_seconds > 0", name="ck_voice_channel_settings_idle_timeout_positive"
        ),
    )

    area_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    voice_channel_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=FALSE, nullable=False
    )
    delegated_task_profile: Mapped[str] = mapped_column(
        String(128),
        default="voice_readonly_v1",
        server_default=text("'voice_readonly_v1'"),
        nullable=False,
    )
    idle_timeout_seconds: Mapped[int] = mapped_column(
        Integer, default=300, server_default=text("300"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=CURRENT_TIMESTAMP, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )


class VoiceSessionRecord(Base):
    """Durable lifecycle envelope for one realtime voice conversation."""

    __tablename__ = "voice_sessions"
    __table_args__ = (
        Index("ix_voice_sessions_owner_started", "owner_person_id", "started_at"),
        CheckConstraint("jsonb_typeof(usage) = 'object'", name="ck_voice_sessions_usage_object"),
        CheckConstraint(
            "(ended_at IS NULL) OR (ended_at >= started_at)",
            name="ck_voice_sessions_ended_after_started",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid4, server_default=GENERATED_UUID
    )
    owner_person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    area_id: Mapped[str] = mapped_column(String(128), nullable=False)
    voice_channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    text_channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_id: Mapped[UUID] = mapped_column(
        ForeignKey("voice_models.id", ondelete="RESTRICT"), nullable=False
    )
    voice_id: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    duplex_mode: Mapped[VoiceDuplexMode] = mapped_column(VOICE_DUPLEX_MODE_ENUM, nullable=False)
    status: Mapped[PersistedVoiceSessionStatus] = mapped_column(
        VOICE_SESSION_STATUS_ENUM,
        default=PersistedVoiceSessionStatus.STARTING,
        server_default=text("'starting'"),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=CURRENT_TIMESTAMP, nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    stop_reason: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    usage: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default=EMPTY_JSONB_OBJECT, nullable=False
    )
    summary: Mapped[str] = mapped_column(
        Text, default="", server_default=EMPTY_STRING, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=CURRENT_TIMESTAMP, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )


class VoiceTurnRecord(Base):
    """One ordered final user or assistant transcript."""

    __tablename__ = "voice_turns"
    __table_args__ = (
        UniqueConstraint("session_id", "sequence"),
        CheckConstraint("sequence > 0", name="ck_voice_turns_sequence_positive"),
        CheckConstraint(
            "char_length(btrim(transcript)) > 0", name="ck_voice_turns_transcript_nonempty"
        ),
        CheckConstraint("jsonb_typeof(usage) = 'object'", name="ck_voice_turns_usage_object"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid4, server_default=GENERATED_UUID
    )
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("voice_sessions.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[VoiceTurnRole] = mapped_column(VOICE_TURN_ROLE_ENUM, nullable=False)
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    provider_item_id: Mapped[str] = mapped_column(
        String(256), default="", server_default=EMPTY_STRING, nullable=False
    )
    usage: Mapped[dict[str, object]] = mapped_column(
        JSONB, default=dict, server_default=EMPTY_JSONB_OBJECT, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=CURRENT_TIMESTAMP, nullable=False
    )


class DelegatedAgentTaskRecord(Base):
    """Durable work envelope submitted by a realtime voice function call."""

    __tablename__ = "delegated_agent_tasks"
    __table_args__ = (
        UniqueConstraint("origin_voice_session_id", "provider_call_id"),
        UniqueConstraint("origin_voice_session_id", "session_sequence"),
        Index("ix_delegated_tasks_owner_created", "owner_person_id", "created_at"),
        Index(
            "ix_delegated_tasks_worker_claim",
            "status",
            "next_attempt_at",
            "created_at",
            postgresql_where=text("status IN ('queued', 'waiting_retry')"),
        ),
        Index(
            "ix_delegated_tasks_mailbox",
            "origin_voice_session_id",
            "notification_state",
            "finished_at",
            postgresql_where=text(
                "status IN ('succeeded', 'failed', 'cancelled', 'interrupted') "
                "AND notification_state IN ('pending', 'deferred')"
            ),
        ),
        CheckConstraint("session_sequence > 0", name="ck_delegated_tasks_sequence_positive"),
        CheckConstraint("retry_count >= 0", name="ck_delegated_tasks_retry_nonnegative"),
        CheckConstraint(
            "char_length(btrim(objective)) BETWEEN 1 AND 4000",
            name="ck_delegated_tasks_objective_length",
        ),
        CheckConstraint(
            "char_length(progress_stage) <= 64 AND char_length(progress_summary) <= 512",
            name="ck_delegated_tasks_progress_length",
        ),
        CheckConstraint(
            "char_length(result_summary) <= 1000 AND char_length(result_text) <= 16000",
            name="ck_delegated_tasks_result_length",
        ),
        CheckConstraint(
            "char_length(error_code) <= 128 AND char_length(error_message) <= 1000",
            name="ck_delegated_tasks_error_length",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid4, server_default=GENERATED_UUID
    )
    owner_person_id: Mapped[str] = mapped_column(String(128), nullable=False)
    area_id: Mapped[str] = mapped_column(String(128), nullable=False)
    text_channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    voice_channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    origin_voice_session_id: Mapped[UUID] = mapped_column(
        ForeignKey("voice_sessions.id", ondelete="RESTRICT"), nullable=False
    )
    session_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    provider_call_id: Mapped[str] = mapped_column(String(256), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    result_style: Mapped[DelegatedResultStyle] = mapped_column(
        DELEGATED_RESULT_STYLE_ENUM,
        default=DelegatedResultStyle.BRIEF,
        server_default=text("'brief'"),
        nullable=False,
    )
    status: Mapped[DelegatedTaskStatus] = mapped_column(
        DELEGATED_TASK_STATUS_ENUM,
        default=DelegatedTaskStatus.QUEUED,
        server_default=text("'queued'"),
        nullable=False,
    )
    lane: Mapped[DelegatedTaskLane] = mapped_column(
        DELEGATED_TASK_LANE_ENUM,
        default=DelegatedTaskLane.READ_PARALLEL,
        server_default=text("'read_parallel'"),
        nullable=False,
    )
    conflict_key: Mapped[str] = mapped_column(
        String(256), default="", server_default=EMPTY_STRING, nullable=False
    )
    notification_state: Mapped[DelegatedTaskNotificationState] = mapped_column(
        TASK_NOTIFICATION_STATE_ENUM,
        default=DelegatedTaskNotificationState.PENDING,
        server_default=text("'pending'"),
        nullable=False,
    )
    agent_model_id: Mapped[UUID] = mapped_column(
        ForeignKey("llm_models.id", ondelete="RESTRICT"), nullable=False
    )
    allowed_tool_names: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, server_default=text("'{}'::text[]"), nullable=False
    )
    agent_thread_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("agent_threads.id", ondelete="SET NULL"), nullable=True
    )
    agent_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True
    )
    progress_stage: Mapped[str] = mapped_column(
        String(64), default="", server_default=EMPTY_STRING, nullable=False
    )
    progress_summary: Mapped[str] = mapped_column(
        String(512), default="", server_default=EMPTY_STRING, nullable=False
    )
    result_summary: Mapped[str] = mapped_column(
        Text, default="", server_default=EMPTY_STRING, nullable=False
    )
    result_text: Mapped[str] = mapped_column(
        Text, default="", server_default=EMPTY_STRING, nullable=False
    )
    error_code: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    error_message: Mapped[str] = mapped_column(
        Text, default="", server_default=EMPTY_STRING, nullable=False
    )
    retry_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default=text("0"), nullable=False
    )
    worker_id: Mapped[str] = mapped_column(
        String(128), default="", server_default=EMPTY_STRING, nullable=False
    )
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    presented_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, server_default=CURRENT_TIMESTAMP, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=CURRENT_TIMESTAMP,
        server_onupdate=FetchedValue(),
        nullable=False,
    )
