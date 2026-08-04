"""Provider-neutral events consumed by the future voice session coordinator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .models import PcmChunk


@dataclass(frozen=True, slots=True)
class VoiceSessionReady:
    """The Provider accepted configuration and can receive audio."""


@dataclass(frozen=True, slots=True)
class VoiceSessionFinished:
    """The Provider acknowledged a graceful session finish."""


@dataclass(frozen=True, slots=True)
class VoiceUserSpeechStarted:
    """Server-side VAD observed the owner starting to speak."""


@dataclass(frozen=True, slots=True)
class VoiceUserSpeechStopped:
    """Server-side VAD observed the owner stopping speech."""


@dataclass(frozen=True, slots=True)
class VoiceResponseStarted:
    """A Provider response acquired a stable identifier and began generation."""

    response_id: str

    def __post_init__(self) -> None:
        _validate_identifier(self.response_id, "response")


@dataclass(frozen=True, slots=True)
class VoiceTranscriptFinal:
    """Final text for one user or assistant turn."""

    role: str
    text: str
    provider_item_id: str = ""
    response_id: str = ""

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"} or not self.text.strip():
            raise ValueError("Final transcript requires a supported role and non-empty text")
        if len(self.provider_item_id) > 256:
            raise ValueError("Provider item identifier is too long")
        if len(self.response_id) > 256:
            raise ValueError("Provider response identifier is too long")


@dataclass(frozen=True, slots=True)
class VoiceAssistantAudio:
    """Incremental assistant audio emitted by the Provider."""

    chunk: PcmChunk
    response_id: str
    provider_item_id: str = ""

    def __post_init__(self) -> None:
        _validate_identifier(self.response_id, "response")
        if len(self.provider_item_id) > 256:
            raise ValueError("Provider item identifier is too long")


@dataclass(frozen=True, slots=True)
class VoiceResponseCompleted:
    """One assistant response reached a terminal boundary."""

    response_id: str
    usage: Mapping[str, int | float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _validate_identifier(self.response_id, "response")


@dataclass(frozen=True, slots=True)
class VoiceResponseCancelled:
    """One response stopped before normal completion and may emit late terminal events."""

    response_id: str
    usage: Mapping[str, int | float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _validate_identifier(self.response_id, "response")


@dataclass(frozen=True, slots=True)
class VoiceProviderFailed:
    """A sanitized typed Provider failure."""

    error_kind: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class VoiceToolCall:
    """One complete Provider function call awaiting a fast control result."""

    call_id: str
    name: str
    arguments: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _validate_identifier(self.call_id, "tool call")
        if not self.name.strip() or len(self.name) > 128:
            raise ValueError("Provider tool name is invalid")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))


VoiceModelEvent = (
    VoiceSessionReady
    | VoiceSessionFinished
    | VoiceUserSpeechStarted
    | VoiceUserSpeechStopped
    | VoiceResponseStarted
    | VoiceTranscriptFinal
    | VoiceAssistantAudio
    | VoiceResponseCompleted
    | VoiceResponseCancelled
    | VoiceToolCall
    | VoiceProviderFailed
)


def _validate_identifier(value: str, kind: str) -> None:
    if not value.strip() or len(value) > 256:
        raise ValueError(f"Provider {kind} identifier is invalid")
