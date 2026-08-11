"""Area-scoped shared playlist use cases."""

from __future__ import annotations

import logging
import unicodedata
from uuid import UUID

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.agent.models import AgentIdentity
from cywl_oopz.settings import MusicSettings

from .errors import (
    MusicAreaRequiredError,
    MusicCatalogError,
    MusicNotFoundError,
    MusicPlaylistEmptyError,
    MusicPlaylistNameError,
    MusicPlaylistNotFoundError,
    NeteasePlaylistIncompleteError,
    NeteasePlaylistTooLargeError,
)
from .models import (
    MusicPlaylist,
    MusicPlaylistEntry,
    MusicPlaylistSummary,
    NeteasePlaylistImport,
    NeteasePlaylistSnapshot,
    PlaylistClear,
    PlaylistDeletion,
    PlaylistQueueLoad,
    PlaylistRename,
    PlaylistTrackRemoval,
)
from .ports import MusicPlaylistRepository, MusicPlaylistSource
from .service import MusicRequestService

logger = logging.getLogger(__name__)


class MusicPlaylistService:
    """Coordinate catalog, durable playlists, and the in-memory playback queue."""

    max_name_characters = 80

    def __init__(
        self,
        settings: MusicSettings,
        repository: MusicPlaylistRepository,
        music: MusicRequestService,
        playlist_source: MusicPlaylistSource | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._music = music
        self._playlist_source = playlist_source

    async def create(self, identity: AgentIdentity, name: str) -> MusicPlaylist:
        area_id = self._area_id(identity)
        display_name, normalized_name = self._normalize_name(name)
        playlist = await self._repository.create(
            area_id,
            display_name,
            normalized_name,
            identity.person_id,
        )
        logger.info(
            "Music playlist created: area=%s playlist=%s",
            opaque_ref(area_id),
            playlist.id,
        )
        return playlist

    async def list(self, identity: AgentIdentity) -> tuple[MusicPlaylistSummary, ...]:
        area_id = self._area_id(identity)
        playlists = await self._repository.list(area_id)
        logger.debug(
            "Music playlists listed: area=%s count=%s",
            opaque_ref(area_id),
            len(playlists),
        )
        return playlists

    async def get(self, identity: AgentIdentity, playlist_id: UUID) -> MusicPlaylist:
        area_id = self._area_id(identity)
        playlist = await self._repository.get(area_id, playlist_id)
        if playlist is None:
            raise MusicPlaylistNotFoundError("Music playlist was not found in this area")
        return playlist

    async def add(
        self,
        identity: AgentIdentity,
        playlist_id: UUID,
        query: str,
    ) -> MusicPlaylistEntry:
        area_id = self._area_id(identity)
        matches = await self._music.search(query, limit=1)
        if not matches:
            raise MusicNotFoundError("No music matched the query")
        entry = await self._repository.append(
            area_id,
            playlist_id,
            matches[0],
            identity.person_id,
            max_tracks=self._settings.max_playlist_tracks,
        )
        logger.info(
            "Music playlist track appended: area=%s playlist=%s entry=%s position=%s",
            opaque_ref(area_id),
            playlist_id,
            entry.id,
            entry.position,
        )
        return entry

    async def remove(
        self,
        identity: AgentIdentity,
        playlist_id: UUID,
        entry_id: UUID,
    ) -> PlaylistTrackRemoval:
        area_id = self._area_id(identity)
        result = await self._repository.remove(area_id, playlist_id, entry_id)
        logger.info(
            "Music playlist track removed: area=%s playlist=%s entry=%s removed=%s",
            opaque_ref(area_id),
            playlist_id,
            entry_id,
            result.removed,
        )
        return result

    async def rename(
        self,
        identity: AgentIdentity,
        playlist_id: UUID,
        name: str,
    ) -> PlaylistRename:
        area_id = self._area_id(identity)
        display_name, normalized_name = self._normalize_name(name)
        result = await self._repository.rename(
            area_id,
            playlist_id,
            display_name,
            normalized_name,
        )
        logger.info(
            "Music playlist renamed: area=%s playlist=%s changed=%s",
            opaque_ref(area_id),
            playlist_id,
            result.changed,
        )
        return result

    async def delete(
        self,
        identity: AgentIdentity,
        playlist_id: UUID,
    ) -> PlaylistDeletion:
        area_id = self._area_id(identity)
        result = await self._repository.delete(area_id, playlist_id)
        logger.info(
            "Music playlist deleted: area=%s playlist=%s deleted=%s removed_tracks=%s",
            opaque_ref(area_id),
            playlist_id,
            result.deleted,
            result.removed_track_count,
        )
        return result

    async def clear(
        self,
        identity: AgentIdentity,
        playlist_id: UUID,
    ) -> PlaylistClear:
        area_id = self._area_id(identity)
        result = await self._repository.clear(area_id, playlist_id)
        logger.info(
            "Music playlist cleared: area=%s playlist=%s removed_tracks=%s",
            opaque_ref(area_id),
            playlist_id,
            result.removed_track_count,
        )
        return result

    async def load(
        self,
        identity: AgentIdentity,
        playlist_id: UUID,
    ) -> PlaylistQueueLoad:
        playlist = await self.get(identity, playlist_id)
        if not playlist.entries:
            raise MusicPlaylistEmptyError("An empty playlist cannot rebuild playback")
        queue = await self._music.replace_queue(
            identity,
            tuple(entry.track for entry in playlist.entries),
        )
        logger.info(
            "Music playlist loaded into queue: area=%s playlist=%s tracks=%s",
            opaque_ref(playlist.area_id),
            playlist.id,
            queue.loaded_count,
        )
        return PlaylistQueueLoad(playlist.id, playlist.name, queue)

    async def preview_netease(self, reference: str) -> NeteasePlaylistSnapshot:
        """Read one bounded source snapshot without mutating the area playlist catalog."""
        if self._playlist_source is None:
            raise MusicCatalogError("Netease playlist import is not configured")
        snapshot = await self._playlist_source.playlist(
            reference,
            limit=self._settings.max_playlist_tracks,
        )
        logger.info(
            "Netease playlist previewed: source=%s declared_tracks=%s visible_tracks=%s "
            "complete=%s",
            snapshot.source_id,
            snapshot.declared_track_count,
            snapshot.loaded_track_count,
            snapshot.complete,
        )
        return snapshot

    async def import_netease(
        self,
        identity: AgentIdentity,
        reference: str,
        *,
        name: str | None,
        allow_partial: bool,
    ) -> NeteasePlaylistImport:
        """Fetch and atomically persist a Netease playlist in the caller's area."""
        area_id = self._area_id(identity)
        snapshot = await self.preview_netease(reference)
        if snapshot.declared_track_count > self._settings.max_playlist_tracks and not allow_partial:
            raise NeteasePlaylistTooLargeError(
                "Netease playlist exceeds the configured area playlist capacity"
            )
        if not snapshot.complete and not allow_partial:
            raise NeteasePlaylistIncompleteError(
                "Netease playlist is incomplete; explicit confirmation is required"
            )
        if not snapshot.tracks:
            raise MusicPlaylistEmptyError("Netease playlist has no importable tracks")
        display_name, normalized_name = self._normalize_name(name or snapshot.name)
        playlist = await self._repository.create_with_tracks(
            area_id,
            display_name,
            normalized_name,
            snapshot.tracks,
            identity.person_id,
            max_tracks=self._settings.max_playlist_tracks,
        )
        result = NeteasePlaylistImport(
            snapshot.source_id,
            snapshot.name,
            snapshot.declared_track_count,
            playlist,
        )
        logger.info(
            "Netease playlist imported: area=%s source=%s playlist=%s imported=%s skipped=%s",
            opaque_ref(area_id),
            snapshot.source_id,
            playlist.id,
            result.imported_track_count,
            result.skipped_track_count,
        )
        return result

    @classmethod
    def _normalize_name(cls, name: str) -> tuple[str, str]:
        display_name = " ".join(unicodedata.normalize("NFKC", name).split())
        if not display_name:
            raise MusicPlaylistNameError("Music playlist name must not be empty")
        normalized_name = display_name.casefold()
        if (
            len(display_name) > cls.max_name_characters
            or len(normalized_name) > cls.max_name_characters
        ):
            raise MusicPlaylistNameError("Music playlist name is too long")
        return display_name, normalized_name

    @staticmethod
    def _area_id(identity: AgentIdentity) -> str:
        area_id = identity.conversation.area_id.strip()
        if not area_id:
            raise MusicAreaRequiredError("Shared music playlists require an OOPZ area")
        return area_id
