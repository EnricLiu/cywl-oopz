"""Provider-neutral values for music search, queue state, and playback."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    RECOVERING = "recovering"
    RELEASING = "releasing"
    FAILED = "failed"


class PlaybackOrder(StrEnum):
    """How one queue cycle selects its remaining tracks."""

    SEQUENTIAL = "sequential"
    SHUFFLE = "shuffle"


class RepeatPolicy(StrEnum):
    """How completed tracks are retained across queue cycles."""

    OFF = "off"
    ONE = "one"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class MusicPlaybackPolicy:
    """Orthogonal playback order and repeat behavior for one active session."""

    order: PlaybackOrder = PlaybackOrder.SEQUENTIAL
    repeat: RepeatPolicy = RepeatPolicy.OFF

    def __post_init__(self) -> None:
        if self.order is PlaybackOrder.SHUFFLE and self.repeat is RepeatPolicy.ONE:
            raise ValueError("Shuffle order cannot be combined with repeat-one")


class MusicPlaybackEndReason(StrEnum):
    """Project-owned terminal result independent from OOPZ SDK enums."""

    FINISHED = "finished"
    STOPPED = "stopped"
    REPLACED = "replaced"
    TRACK_ERROR = "track_error"
    VOICE_LEFT = "voice_left"
    BACKEND_CLOSED = "backend_closed"


class MusicFailureCode(StrEnum):
    """Stable failure category safe to expose through queue snapshots."""

    TRACK_ERROR = "track_error"
    CATALOG_ERROR = "catalog_error"
    VOICE_LEFT = "voice_left"
    BACKEND_CLOSED = "backend_closed"
    RELEASE_FAILED = "release_failed"


class MusicFailureScope(StrEnum):
    """Subsystem that must recover before playback can progress."""

    TRACK = "track"
    CATALOG = "catalog"
    VOICE_SESSION = "voice_session"


@dataclass(frozen=True, slots=True)
class MusicFailure:
    """Bounded historical failure independent from the current playback phase."""

    code: MusicFailureCode
    scope: MusicFailureScope
    recoverable: bool
    track_id: UUID | None
    retry_count: int
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class MusicPlaybackResult:
    """One complete playback outcome returned by the voice gateway."""

    end_reason: MusicPlaybackEndReason
    duration_seconds: float | None = None
    terminal_error: BaseException | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class MusicQueueSnapshot:
    """Bounded immutable view of one voice channel queue."""

    voice_channel: VoiceChannelKey
    state: PlaybackState
    policy: MusicPlaybackPolicy
    current: QueuedTrack | None
    upcoming: tuple[QueuedTrack, ...]
    cycle_completed_count: int
    revision: int
    last_failure: MusicFailure | None = None


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    """Result returned once a queue mutation is committed."""

    voice_channel: VoiceChannelKey
    item: QueuedTrack
    position: int
    started_worker: bool


@dataclass(frozen=True, slots=True)
class PlaybackPolicyChange:
    """Result of changing one voice channel's playback policy."""

    voice_channel: VoiceChannelKey
    policy: MusicPlaybackPolicy
    changed: bool


@dataclass(frozen=True, slots=True)
class QueueRebuildResult:
    """Result of replacing one voice channel's active and upcoming queue."""

    voice_channel: VoiceChannelKey
    loaded_count: int
    replaced_current: bool
    started_worker: bool


@dataclass(frozen=True, slots=True)
class MusicQueueClearResult:
    """Result of stopping and clearing one transient playback session."""

    voice_channel: VoiceChannelKey
    stopped_current: bool
    removed_count: int


@dataclass(frozen=True, slots=True)
class MusicPlaylistEntry:
    """One stable, ordered track snapshot stored in a shared area playlist."""

    id: UUID
    playlist_id: UUID
    position: int
    track: MusicTrack
    added_by_person_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MusicPlaylist:
    """An area-scoped shared playlist and its ordered entries."""

    id: UUID
    area_id: str
    name: str
    normalized_name: str
    created_by_person_id: str
    entries: tuple[MusicPlaylistEntry, ...]
    created_at: datetime
    updated_at: datetime

    @property
    def track_count(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, slots=True)
class MusicPlaylistSummary:
    """Compact playlist metadata for area-level discovery."""

    id: UUID
    area_id: str
    name: str
    track_count: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PlaylistTrackRemoval:
    """Result of deleting and compacting one playlist entry."""

    playlist_id: UUID
    entry_id: UUID
    removed: bool


@dataclass(frozen=True, slots=True)
class PlaylistRename:
    """Idempotent area-scoped playlist rename result."""

    playlist_id: UUID
    old_name: str
    new_name: str
    changed: bool


@dataclass(frozen=True, slots=True)
class PlaylistDeletion:
    """Idempotent playlist deletion and cascaded track count."""

    playlist_id: UUID
    name: str | None
    deleted: bool
    removed_track_count: int


@dataclass(frozen=True, slots=True)
class PlaylistClear:
    """Track deletion result that preserves the shared playlist."""

    playlist_id: UUID
    name: str
    removed_track_count: int


@dataclass(frozen=True, slots=True)
class PlaylistQueueLoad:
    """A playlist plus the queue replacement committed from its entries."""

    playlist_id: UUID
    playlist_name: str
    queue: QueueRebuildResult


@dataclass(frozen=True, slots=True)
class NeteasePlaylistSnapshot:
    """Bounded Netease playlist metadata and the tracks visible to the bot."""

    source_id: str
    name: str
    declared_track_count: int
    tracks: tuple[MusicTrack, ...]

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.name.strip():
            raise ValueError("Netease playlist ID and name must not be empty")
        if self.declared_track_count < 0:
            raise ValueError("Netease playlist track count must not be negative")
        if any(track.source != "netease" for track in self.tracks):
            raise ValueError("Netease playlist tracks must use the Netease source")

    @property
    def loaded_track_count(self) -> int:
        return len(self.tracks)

    @property
    def complete(self) -> bool:
        return self.loaded_track_count >= self.declared_track_count


@dataclass(frozen=True, slots=True)
class NeteasePlaylistImport:
    """Area playlist created from one bounded Netease snapshot."""

    source_id: str
    source_name: str
    declared_track_count: int
    playlist: MusicPlaylist

    @property
    def imported_track_count(self) -> int:
        return self.playlist.track_count

    @property
    def skipped_track_count(self) -> int:
        return max(0, self.declared_track_count - self.imported_track_count)

    @property
    def partial(self) -> bool:
        return self.skipped_track_count > 0
