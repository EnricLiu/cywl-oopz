"""Framework-neutral values shared by administration use cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID


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


@dataclass(frozen=True, slots=True)
class OopzMessageAddress:
    """Transport-complete address used to bind and verify outbound messages."""

    scope: OopzMessageScope
    area_id: str
    channel_id: str
    target_person_id: str = ""

    def __post_init__(self) -> None:
        area_id = self.area_id.strip()
        channel_id = _identifier(self.channel_id, "Message channel ID")
        target = self.target_person_id.strip()
        if max(len(area_id), len(channel_id), len(target)) > 128:
            raise ValueError("Message address identifiers must be at most 128 characters")
        if self.scope is OopzMessageScope.CHANNEL:
            if not area_id or target:
                raise ValueError("Channel message addresses require only area and channel")
        elif area_id or not target:
            raise ValueError("Private message addresses require channel and target person")
        object.__setattr__(self, "area_id", area_id)
        object.__setattr__(self, "channel_id", channel_id)
        object.__setattr__(self, "target_person_id", target)


@dataclass(frozen=True, slots=True)
class OutboundMessageReceipt:
    """Durable ownership and diagnostic linkage for one Bot message."""

    message_id: str
    message_timestamp: str
    kind: OutboundMessageKind
    state: OutboundMessageState
    address: OopzMessageAddress
    in_reply_to_message_id: str = ""
    owner_person_id: str = ""
    agent_run_id: UUID | None = None
    diagnostic_snapshot: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        message_id = self.message_id.strip()
        timestamp = self.message_timestamp.strip()
        in_reply_to = self.in_reply_to_message_id.strip()
        owner = self.owner_person_id.strip()
        if not message_id or len(message_id) > 256:
            raise ValueError("Outbound message ID must contain at most 256 characters")
        if len(timestamp) > 64 or len(in_reply_to) > 256 or len(owner) > 128:
            raise ValueError("Outbound message metadata exceeds its storage limit")
        object.__setattr__(self, "message_id", message_id)
        object.__setattr__(self, "message_timestamp", timestamp)
        object.__setattr__(self, "in_reply_to_message_id", in_reply_to)
        object.__setattr__(self, "owner_person_id", owner)
        object.__setattr__(
            self,
            "diagnostic_snapshot",
            MappingProxyType(dict(self.diagnostic_snapshot)),
        )


@dataclass(frozen=True, slots=True)
class AgentDiagnosticTool:
    """One persisted tool execution projected for bounded diagnostics."""

    call_id: str
    name: str
    version: str
    effect: str
    status: str
    input_payload: Mapping[str, Any]
    output_payload: Mapping[str, Any] | None
    error_code: str
    started_at: datetime
    finished_at: datetime | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_payload", MappingProxyType(dict(self.input_payload)))
        if self.output_payload is not None:
            object.__setattr__(
                self,
                "output_payload",
                MappingProxyType(dict(self.output_payload)),
            )


@dataclass(frozen=True, slots=True)
class AgentResponseDiagnostic:
    """Immutable aggregate behind one tracked Agent response."""

    receipt: OutboundMessageReceipt
    run_id: UUID | None = None
    thread_id: UUID | None = None
    status: str = ""
    stop_reason: str = ""
    error_code: str = ""
    provider_alias: str = ""
    model_alias: str = ""
    selection_source: str = ""
    limits: Mapping[str, Any] = MappingProxyType({})
    usage: Mapping[str, Any] = MappingProxyType({})
    run_diagnostics: Mapping[str, Any] = MappingProxyType({})
    started_at: datetime | None = None
    finished_at: datetime | None = None
    assistant_text: str = ""
    tools: tuple[AgentDiagnosticTool, ...] = ()

    def __post_init__(self) -> None:
        for name in ("limits", "usage", "run_diagnostics"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
