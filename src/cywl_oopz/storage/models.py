"""SQLAlchemy models owned by the CYWL application."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Integer, String, Text, UniqueConstraint
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
