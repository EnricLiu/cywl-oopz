"""Framework-neutral values shared by administration use cases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OopzMessageScope(StrEnum):
    CHANNEL = "channel"
    PRIVATE = "private"


class OutboundMessageKind(StrEnum):
    AGENT_RESPONSE = "agent_response"
    COMMAND_REPLY = "command_reply"
    STATUS = "status"
    NOTIFICATION = "notification"


class OutboundMessageState(StrEnum):
    ACTIVE = "active"
    FINAL = "final"
    RECALLED = "recalled"
    SUPERSEDED = "superseded"


def _identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) > 128:
        raise ValueError(f"{label} must be at most 128 characters")
    return normalized


@dataclass(frozen=True, slots=True)
class ChannelKey:
    """One project-owned OOPZ channel address."""

    area_id: str
    channel_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "area_id", _identifier(self.area_id, "Area ID"))
        object.__setattr__(self, "channel_id", _identifier(self.channel_id, "Channel ID"))


@dataclass(frozen=True, slots=True)
class AreaChannelCatalog:
    """Visible text and voice channels discovered for one area."""

    area_id: str
    text_channels: tuple[ChannelKey, ...] = ()
    voice_channels: tuple[ChannelKey, ...] = ()

    def __post_init__(self) -> None:
        area_id = _identifier(self.area_id, "Area ID")
        object.__setattr__(self, "area_id", area_id)
        all_keys = (*self.text_channels, *self.voice_channels)
        if any(key.area_id != area_id for key in all_keys):
            raise ValueError("Catalog channels must belong to its area")
        identities = [key.channel_id for key in all_keys]
        if len(identities) != len(set(identities)):
            raise ValueError("Catalog channel IDs must be unique")


@dataclass(frozen=True, slots=True)
class ChannelInitializationResult:
    """Outcome of idempotently initializing one text channel."""

    created: bool


@dataclass(frozen=True, slots=True)
class AreaInitializationResult:
    """Created/existing counts for one area initialization transaction."""

    text_created: int
    text_existing: int
    voice_created: int
    voice_existing: int

    def __post_init__(self) -> None:
        if (
            min(
                self.text_created,
                self.text_existing,
                self.voice_created,
                self.voice_existing,
            )
            < 0
        ):
            raise ValueError("Initialization counts must not be negative")
