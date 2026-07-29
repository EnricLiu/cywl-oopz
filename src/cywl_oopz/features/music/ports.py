"""Replaceable catalog and OOPZ voice boundaries for music playback."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from .models import (
    MusicPlaylist,
    MusicPlaylistEntry,
    MusicPlaylistSummary,
    MusicTrack,
    NeteasePlaylistSnapshot,
    PlayableTrack,
    PlaylistTrackRemoval,
    VoiceChannelKey,
)


class MusicCatalog(Protocol):
    """Search metadata and resolve expiring stream URLs."""

    async def search(self, query: str, *, limit: int) -> tuple[MusicTrack, ...]:
        """Return a bounded ordered result set."""

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        """Resolve a playable URL immediately before playback."""

    async def aclose(self) -> None:
        """Release owned HTTP resources."""


class MusicPlaylistSource(Protocol):
    """Read bounded external playlists without exposing provider responses."""

    async def playlist(
        self,
        reference: str,
        *,
        limit: int,
    ) -> NeteasePlaylistSnapshot:
        """Return metadata and up to ``limit`` visible tracks."""


class MusicVoiceGateway(Protocol):
    """OOPZ-independent voice operations required by the queue worker."""

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        """Return the user's current voice channel in an area."""

    async def play(self, channel: VoiceChannelKey, stream_url: str) -> None:
        """Join the channel if needed and begin URL playback."""

    async def state(self) -> str:
        """Return the current backend playback state."""

    async def stop(self) -> None:
        """Stop the current track."""

    async def pause(self) -> bool:
        """Pause the current track."""

    async def resume(self) -> bool:
        """Resume the current track."""

    async def leave(self, channel: VoiceChannelKey) -> bool:
        """Leave the channel only when it still owns the shared voice backend."""

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
