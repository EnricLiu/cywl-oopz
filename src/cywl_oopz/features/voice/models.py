"""Provider- and SDK-neutral values for realtime voice conversations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from uuid import UUID


@dataclass(frozen=True, slots=True)
class VoiceChannelKey:
    """One OOPZ area voice channel without importing SDK models."""

    area_id: str
    channel_id: str

    def __post_init__(self) -> None:
        if not self.area_id.strip() or not self.channel_id.strip():
            raise ValueError("Voice channel area and channel identifiers must not be empty")


@dataclass(frozen=True, slots=True)
class VoiceTextAddress:
    """Text channel where the user started the voice conversation."""

    area_id: str
    channel_id: str

    def __post_init__(self) -> None:
        if not self.area_id.strip() or not self.channel_id.strip():
            raise ValueError("Voice origin area and channel identifiers must not be empty")


@dataclass(frozen=True, slots=True)
class VoiceAudioFormat:
    """PCM shape accepted at the project boundary."""

    sample_rate: int
    channels: int
    sample_format: Literal["s16le", "f32le"]

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("Voice audio sample rate must be positive")
        if self.channels <= 0:
            raise ValueError("Voice audio channel count must be positive")
        if self.sample_format not in {"s16le", "f32le"}:
            raise ValueError("Voice audio sample format is unsupported")

    @property
    def sample_width_bytes(self) -> int:
        return 2 if self.sample_format == "s16le" else 4

    @property
    def frame_width_bytes(self) -> int:
        return self.channels * self.sample_width_bytes


@dataclass(frozen=True, slots=True)
class RemoteAudioFrame:
    """One timestamped PCM frame received from the owner audio track."""

    pcm: bytes
    format: VoiceAudioFormat
    sequence: int
    captured_at_monotonic: float
    source_dropped_frames: int = 0

    def __post_init__(self) -> None:
        if not self.pcm or len(self.pcm) % self.format.frame_width_bytes:
            raise ValueError("Remote audio frame must contain aligned PCM samples")
        if self.sequence < 0 or self.captured_at_monotonic < 0:
            raise ValueError("Remote audio sequence and timestamp must not be negative")
        if self.source_dropped_frames < 0:
            raise ValueError("Dropped frame count must not be negative")


@dataclass(frozen=True, slots=True)
class PcmChunk:
    """One bounded PCM chunk tagged with the current response generation."""

    pcm: bytes
    format: VoiceAudioFormat
    duration_ms: int
    generation: int

    def __post_init__(self) -> None:
        if not self.pcm or len(self.pcm) % self.format.frame_width_bytes:
            raise ValueError("PCM chunk must contain aligned samples")
        if self.duration_ms <= 0 or self.generation < 0:
            raise ValueError("PCM duration must be positive and generation non-negative")
        sample_count = len(self.pcm) // self.format.frame_width_bytes
        duration_error = abs(sample_count * 1000 - self.format.sample_rate * self.duration_ms)
        if duration_error * 2 > self.format.sample_rate:
            raise ValueError("PCM byte length must match its rounded duration")


@dataclass(frozen=True, slots=True)
class PlaybackCursor:
    """Provider-neutral output position used for precise interruption."""

    generation: int
    accepted_samples: int
    rendered_samples: int
    buffered_samples: int
    sample_rate: int

    def __post_init__(self) -> None:
        if self.generation < 0 or self.sample_rate <= 0:
            raise ValueError("Playback cursor generation and sample rate are invalid")
        if min(self.accepted_samples, self.rendered_samples, self.buffered_samples) < 0:
            raise ValueError("Playback cursor sample counts must not be negative")
        if self.rendered_samples > self.accepted_samples:
            raise ValueError("Rendered samples cannot exceed accepted samples")
        if self.rendered_samples + self.buffered_samples > self.accepted_samples:
            raise ValueError("Rendered and buffered samples cannot exceed accepted samples")

    @property
    def rendered_ms(self) -> int:
        """Return the real playout position rounded down to milliseconds."""
        return self.rendered_samples * 1000 // self.sample_rate


class VoiceSessionState(StrEnum):
    """Observable in-memory state of the active conversation."""

    STARTING = "starting"
    ACQUIRING_VOICE = "acquiring_voice"
    RESOLVING_SPEAKER = "resolving_speaker"
    CONNECTING_PROVIDER = "connecting_provider"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTING = "interrupting"
    RECOVERING = "recovering"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


class VoiceStopReason(StrEnum):
    """Stable reason supplied to a session runtime during teardown."""

    COMMAND = "command"
    SHUTDOWN = "shutdown"
    RUNTIME_ENDED = "runtime_ended"
    START_FAILED = "start_failed"


class VoiceMediaEndReason(StrEnum):
    """Project-owned terminal reason for the owner input track."""

    CLOSED_BY_CALLER = "closed_by_caller"
    OWNER_UNPUBLISHED = "owner_unpublished"
    OWNER_LEFT = "owner_left"
    VOICE_LEFT = "voice_left"
    BACKEND_CLOSED = "backend_closed"
    TRANSPORT_LOST = "transport_lost"
    QUEUE_OVERFLOW = "queue_overflow"


@dataclass(frozen=True, slots=True)
class VoiceMediaTerminal:
    """Sanitized terminal state surfaced by the OOPZ media adapter."""

    reason: VoiceMediaEndReason
    error_kind: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceStartRequest:
    """Trusted identity and reply address extracted from an OOPZ command."""

    owner_person_id: str
    origin: VoiceTextAddress

    def __post_init__(self) -> None:
        if not self.owner_person_id.strip():
            raise ValueError("Voice session owner must not be empty")


@dataclass(frozen=True, slots=True)
class VoiceSessionDescriptor:
    """Pinned identity of one in-memory conversation runtime."""

    session_id: UUID
    owner_person_id: str
    voice_channel: VoiceChannelKey
    origin: VoiceTextAddress


@dataclass(frozen=True, slots=True)
class VoiceSessionStatus:
    """Credential-free snapshot rendered by commands and presenters."""

    active: bool
    session_id: UUID | None = None
    owner_person_id: str = ""
    voice_channel: VoiceChannelKey | None = None
    state: VoiceSessionState = VoiceSessionState.CLOSED
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.elapsed_seconds < 0:
            raise ValueError("Voice session elapsed time must not be negative")
        if self.active and (self.session_id is None or not self.owner_person_id):
            raise ValueError("Active voice status requires session identity and owner")


@dataclass(frozen=True, slots=True)
class VoiceRuntimeResult:
    """Terminal result returned by a session runtime."""

    reason: VoiceStopReason


@dataclass(frozen=True, slots=True)
class VoiceProviderCapabilities:
    """Explicit optional operations supported by a realtime Provider."""

    tool_calls: bool = False
    context_injection: bool = False
    proactive_response: bool = False
    external_text_speech: bool = False
