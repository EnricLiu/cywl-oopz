"""Provider-neutral values for music search, queue state, and playback."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class VoiceChannelKey:
    """Stable state boundary for one OOPZ voice channel."""

    area_id: str
    channel_id: str

    def __post_init__(self) -> None:
        if not self.area_id.strip() or not self.channel_id.strip():
            raise ValueError("Voice channel area and ID must not be empty")


@dataclass(frozen=True, slots=True)
class MusicTrack:
    """Catalog metadata safe to persist in an in-memory queue or expose to a model."""

    source: str
    source_id: str
    title: str
    artists: tuple[str, ...]
    duration_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.source_id.strip() or not self.title.strip():
            raise ValueError("Music source, source ID, and title must not be empty")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("Music duration must not be negative")


@dataclass(frozen=True, slots=True)
class PlayableTrack:
    """A catalog track with a short-lived stream URL resolved at playback time."""

    track: MusicTrack
    stream_url: str

    def __post_init__(self) -> None:
        if not self.stream_url.strip():
            raise ValueError("Music stream URL must not be empty")


@dataclass(frozen=True, slots=True)
class QueuedTrack:
    """One enqueue request with a stable identity."""

    track: MusicTrack
    requested_by: str
    id: UUID = field(default_factory=uuid4)


class PlaybackState(StrEnum):
    """Project-owned state independent from browser backend strings."""

    IDLE = "idle"
    WAITING = "waiting"
    LOADING = "loading"
    PLAYING = "playing"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MusicQueueSnapshot:
    """Bounded immutable view of one voice channel queue."""

    voice_channel: VoiceChannelKey
    state: PlaybackState
    current: QueuedTrack | None
    upcoming: tuple[QueuedTrack, ...]
    revision: int


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    """Result returned once a queue mutation is committed."""

    voice_channel: VoiceChannelKey
    item: QueuedTrack
    position: int
    started_worker: bool
