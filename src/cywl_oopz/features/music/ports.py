"""Replaceable catalog and OOPZ voice boundaries for music playback."""

from __future__ import annotations

from typing import Protocol

from .models import MusicTrack, PlayableTrack, VoiceChannelKey


class MusicCatalog(Protocol):
    """Search metadata and resolve expiring stream URLs."""

    async def search(self, query: str, *, limit: int) -> tuple[MusicTrack, ...]:
        """Return a bounded ordered result set."""

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        """Resolve a playable URL immediately before playback."""

    async def aclose(self) -> None:
        """Release owned HTTP resources."""


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

    async def aclose(self) -> None:
        """Stop playback and leave the active voice channel."""
