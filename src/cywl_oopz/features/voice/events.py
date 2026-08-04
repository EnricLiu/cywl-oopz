"""Provider-neutral events consumed by the future voice session coordinator."""

from __future__ import annotations

from dataclasses import dataclass

from .models import PcmChunk


@dataclass(frozen=True, slots=True)
class VoiceSessionReady:
    """The Provider accepted configuration and can receive audio."""


@dataclass(frozen=True, slots=True)
class VoiceUserSpeechStarted:
    """Server-side VAD observed the owner starting to speak."""


@dataclass(frozen=True, slots=True)
class VoiceUserSpeechStopped:
    """Server-side VAD observed the owner stopping speech."""


@dataclass(frozen=True, slots=True)
class VoiceTranscriptFinal:
    """Final text for one user or assistant turn."""

    role: str
    text: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"} or not self.text.strip():
            raise ValueError("Final transcript requires a supported role and non-empty text")


@dataclass(frozen=True, slots=True)
class VoiceAssistantAudio:
    """Incremental assistant audio emitted by the Provider."""

    chunk: PcmChunk


@dataclass(frozen=True, slots=True)
class VoiceResponseCompleted:
    """One assistant response reached a terminal boundary."""

    response_id: str


@dataclass(frozen=True, slots=True)
class VoiceProviderFailed:
    """A sanitized typed Provider failure."""

    error_kind: str
    retryable: bool


VoiceModelEvent = (
    VoiceSessionReady
    | VoiceUserSpeechStarted
    | VoiceUserSpeechStopped
    | VoiceTranscriptFinal
    | VoiceAssistantAudio
    | VoiceResponseCompleted
    | VoiceProviderFailed
)
