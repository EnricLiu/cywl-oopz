"""Replaceable catalog and OOPZ voice boundaries for music playback."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from .models import (
    MusicPageLocator,
    MusicPlaybackResult,
    MusicPlaylist,
    MusicPlaylistEntry,
    MusicPlaylistSummary,
    MusicProviderHealth,
    MusicSourceKind,
    MusicTrack,
    MusicTrackReference,
    NeteasePlaylistSnapshot,
    PlayableTrack,
    PlaylistClear,
    PlaylistDeletion,
    PlaylistRename,
    PlaylistTrackRemoval,
    VoiceChannelKey,
)


class MusicProvider(Protocol):
    """One source-specific metadata and playback adapter."""

    @property
    def source(self) -> MusicSourceKind:
        """Return the provider identifier persisted with every track."""

    async def search(self, query: str, *, limit: int) -> tuple[MusicTrack, ...]:
        """Return a bounded ordered result set."""

    async def lookup(self, reference: MusicTrackReference) -> MusicTrack:
        """Load trusted metadata for one normalized stable reference."""

    async def inspect(self, locator: MusicPageLocator) -> MusicTrack:
        """Normalize one allowed page locator into a trusted track snapshot."""

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        """Resolve a playable URL immediately before playback."""

    async def health(self) -> MusicProviderHealth:
        """Return current provider readiness without exposing credentials."""

    async def aclose(self) -> None:
        """Release owned HTTP resources."""


class MusicCatalog(Protocol):
    """Route provider-neutral music use cases to configured sources."""

    async def search(
        self,
        query: str,
        *,
        limit: int,
        source: MusicSourceKind | None = None,
    ) -> tuple[MusicTrack, ...]:
        """Search one explicit source or the configured default."""

    async def lookup(self, reference: MusicTrackReference) -> MusicTrack:
        """Load one stable reference through its owning provider."""

    async def inspect(self, locator: MusicPageLocator) -> MusicTrack:
        """Normalize one allowed page locator through its owning provider."""

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        """Resolve one track through its owning provider."""

    async def health(self) -> tuple[MusicProviderHealth, ...]:
        """Return health for all enabled providers in declaration order."""

    async def aclose(self) -> None:
        """Close all providers exactly once."""


class MusicPlaylistSource(Protocol):
    """Read bounded external playlists without exposing provider responses."""

    async def playlist(
        self,
        reference: str,
        *,
        limit: int,
    ) -> NeteasePlaylistSnapshot:
        """Return metadata and up to ``limit`` visible tracks."""


class MusicPlayback(Protocol):
    """Owner handle for one complete track playback."""

    async def wait_finished(self) -> MusicPlaybackResult:
        """Wait for an authoritative terminal playback event."""

    async def stop(self) -> None:
        """Stop this playback without affecting a replacement owner."""

    async def pause(self) -> bool:
        """Pause this playback when it is still active."""

    async def resume(self) -> bool:
        """Resume this playback when it is still active."""


class MusicVoiceGateway(Protocol):
    """OOPZ-independent lease and playback operations used by music."""

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        """Return the user's current voice channel in an area."""

    async def acquire(self, channel: VoiceChannelKey) -> bool:
        """Reserve the shared backend for this queue without preempting another feature."""

    async def start_playback(
        self,
        channel: VoiceChannelKey,
        playable: PlayableTrack,
    ) -> MusicPlayback:
        """Begin one typed playback under an existing music lease."""

    async def release(self, channel: VoiceChannelKey) -> bool:
        """Release only the matching music lease after its queue drains."""

    async def reset(self, channel: VoiceChannelKey) -> None:
        """Invalidate a stale physical voice generation before a fresh acquire."""

    async def aclose(self) -> None:
        """Stop playback and leave the active voice channel."""


class MusicPlaylistRepository(Protocol):
    """Persistence boundary for area-shared playlists."""

    async def create(
        self,
        area_id: str,
        name: str,
        normalized_name: str,
        created_by_person_id: str,
    ) -> MusicPlaylist:
        """Create one empty playlist."""

    async def list(self, area_id: str) -> tuple[MusicPlaylistSummary, ...]:
        """List compact playlist metadata for exactly one area."""

    async def get(self, area_id: str, playlist_id: UUID) -> MusicPlaylist | None:
        """Load one playlist and its ordered entries."""

    async def append(
        self,
        area_id: str,
        playlist_id: UUID,
        track: MusicTrack,
        added_by_person_id: str,
        *,
        max_tracks: int,
    ) -> MusicPlaylistEntry:
        """Append one track while locking the owning playlist."""

    async def create_with_tracks(
        self,
        area_id: str,
        name: str,
        normalized_name: str,
        tracks: tuple[MusicTrack, ...],
        created_by_person_id: str,
        *,
        max_tracks: int,
    ) -> MusicPlaylist:
        """Atomically create one playlist and all ordered track snapshots."""

    async def remove(
        self,
        area_id: str,
        playlist_id: UUID,
        entry_id: UUID,
    ) -> PlaylistTrackRemoval:
        """Delete one entry and compact following positions."""

    async def rename(
        self,
        area_id: str,
        playlist_id: UUID,
        name: str,
        normalized_name: str,
    ) -> PlaylistRename:
        """Rename one playlist under its area-scoped uniqueness constraint."""

    async def delete(self, area_id: str, playlist_id: UUID) -> PlaylistDeletion:
        """Delete one playlist and its entries, or report that it is already absent."""

    async def clear(self, area_id: str, playlist_id: UUID) -> PlaylistClear:
        """Delete every entry while preserving the playlist."""
