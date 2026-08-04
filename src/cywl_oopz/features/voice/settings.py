"""Immutable PostgreSQL-backed configuration for realtime voice sessions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID

from .models import VoiceChannelKey


class VoiceProviderProtocol(StrEnum):
    QWEN_OMNI_REALTIME_WS = "qwen_omni_realtime_ws"
    QWEN_AUDIO_REALTIME_WS = "qwen_audio_realtime_ws"
    VOLC_REALTIME_DIALOGUE_WS = "volc_realtime_dialogue_ws"


class VoiceModelMode(StrEnum):
    NATIVE_REALTIME = "native_realtime"
    AGENT_CASCADE = "agent_cascade"


class VoiceDuplexMode(StrEnum):
    FULL = "full"
    HALF = "half"


class PersistedVoiceSessionStatus(StrEnum):
    STARTING = "starting"
    ACTIVE = "active"
    RECOVERING = "recovering"
    ENDED = "ended"
    FAILED = "failed"


class VoiceTurnRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class VoiceProviderConfiguration:
    id: UUID
    alias: str
    display_name: str
    protocol: VoiceProviderProtocol
    endpoint: str
    credentials: Mapping[str, Any] = field(repr=False)
    config: Mapping[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class VoiceModelConfiguration:
    id: UUID
    provider_id: UUID
    alias: str
    remote_model_name: str
    display_name: str
    mode: VoiceModelMode
    capabilities: Mapping[str, Any]
    audio_config: Mapping[str, Any]
    prompt_config: Mapping[str, Any]
    limits: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VoiceChannelConfiguration:
    channel: VoiceChannelKey
    delegated_task_profile: str
    idle_timeout_seconds: int


@dataclass(frozen=True, slots=True)
class VoiceStartConfiguration:
    """Fresh-read selection pinned for the lifetime of one session."""

    provider: VoiceProviderConfiguration
    model: VoiceModelConfiguration
    channel: VoiceChannelConfiguration
    voice_id: str
    duplex_mode: VoiceDuplexMode
    delegated_agent_model_id: UUID | None


@dataclass(frozen=True, slots=True)
class SelectableVoiceModel:
    id: UUID
    provider_alias: str
    model_alias: str
    display_name: str
    mode: VoiceModelMode
    selected: bool = False

    @property
    def selector(self) -> str:
        return f"{self.provider_alias}/{self.model_alias}"


@dataclass(frozen=True, slots=True)
class VoiceUserSelection:
    preferred_model_id: UUID | None = None
    voice_id: str = ""
    duplex_mode: VoiceDuplexMode = VoiceDuplexMode.FULL
    delegated_agent_model_id: UUID | None = None
