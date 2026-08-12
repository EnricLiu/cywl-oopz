from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from cywl_oopz.features.agent.models import AgentIdentity, AgentRunLimits
from cywl_oopz.features.agent.tools.builtin import EmptyToolInput
from cywl_oopz.features.agent.tools.models import ToolExecutionContext, ToolExecutionError
from cywl_oopz.features.agent.tools.playlists import (
    AddMusicPlaylistTrackInput,
    AddMusicPlaylistTrackTool,
    ClearMusicPlaylistTool,
    CreateMusicPlaylistInput,
    CreateMusicPlaylistTool,
    DeleteMusicPlaylistTool,
    GetMusicPlaylistTool,
    ImportNeteasePlaylistInput,
    ImportNeteasePlaylistTool,
    ListMusicPlaylistsTool,
    LoadMusicPlaylistTool,
    NeteasePlaylistReferenceInput,
    PlaylistIdInput,
    PreviewNeteasePlaylistTool,
    RemoveMusicPlaylistTrackInput,
    RemoveMusicPlaylistTrackTool,
    RenameMusicPlaylistInput,
    RenameMusicPlaylistTool,
)
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.music.errors import (
    MusicAreaRequiredError,
    MusicPlaylistConflictError,
    MusicPlaylistEmptyError,
    MusicPlaylistFullError,
    MusicPlaylistNotFoundError,
    NeteasePlaylistIncompleteError,
    NeteasePlaylistReferenceError,
    NeteasePlaylistTooLargeError,
)
from cywl_oopz.features.music.models import (
    MusicPlaylist,
    MusicPlaylistEntry,
    MusicPlaylistSummary,
    MusicSourceKind,
    MusicTrack,
    MusicTrackReference,
    NeteasePlaylistSnapshot,
    PlaylistClear,
    PlaylistDeletion,
    PlaylistRename,
    PlaylistTrackRemoval,
    QueueRebuildResult,
    VoiceChannelKey,
)
from cywl_oopz.features.music.playlists import MusicPlaylistService
from cywl_oopz.settings import MusicSettings


def settings() -> MusicSettings:
    return MusicSettings.from_mapping(
        {
            "CYWL_MUSIC_ENABLED": "true",
            "CYWL_MUSIC_CATALOG_BASE_URL": "https://music.example",
            "CYWL_MUSIC_MAX_QUEUE_LENGTH": "3",
            "CYWL_MUSIC_MAX_PLAYLIST_TRACKS": "3",
        }
    )


def identity(
    person_id: str = "person",
    *,
    area_id: str = "area",
) -> AgentIdentity:
    scope = "channel" if area_id else "private"
    return AgentIdentity(
        person_id,
        ConversationKey(scope, area_id, "text" if area_id else "", person_id),
    )


class InMemoryPlaylistRepository:
    def __init__(self) -> None:
        self.playlists: dict[UUID, MusicPlaylist] = {}

    async def create(
        self,
        area_id: str,
        name: str,
        normalized_name: str,
        created_by_person_id: str,
    ) -> MusicPlaylist:
        if any(
            item.area_id == area_id and item.normalized_name == normalized_name
            for item in self.playlists.values()
        ):
            raise MusicPlaylistConflictError
        now = datetime.now(UTC)
        playlist = MusicPlaylist(
            uuid4(),
            area_id,
            name,
            normalized_name,
            created_by_person_id,
            (),
            now,
            now,
        )
        self.playlists[playlist.id] = playlist
        return playlist

    async def list(self, area_id: str) -> tuple[MusicPlaylistSummary, ...]:
        return tuple(
            MusicPlaylistSummary(
                item.id,
                item.area_id,
                item.name,
                item.track_count,
                item.updated_at,
            )
            for item in self.playlists.values()
            if item.area_id == area_id
        )

    async def get(self, area_id: str, playlist_id: UUID) -> MusicPlaylist | None:
        playlist = self.playlists.get(playlist_id)
        return playlist if playlist is not None and playlist.area_id == area_id else None

    async def append(
        self,
        area_id: str,
        playlist_id: UUID,
        track: MusicTrack,
        added_by_person_id: str,
        *,
        max_tracks: int,
    ) -> MusicPlaylistEntry:
        playlist = await self.get(area_id, playlist_id)
        if playlist is None:
            raise MusicPlaylistNotFoundError
        if playlist.track_count >= max_tracks:
            raise MusicPlaylistFullError
        now = datetime.now(UTC)
        entry = MusicPlaylistEntry(
            uuid4(),
            playlist_id,
            playlist.track_count + 1,
            track,
            added_by_person_id,
            now,
        )
        self.playlists[playlist_id] = MusicPlaylist(
            playlist.id,
            playlist.area_id,
            playlist.name,
            playlist.normalized_name,
            playlist.created_by_person_id,
            (*playlist.entries, entry),
            playlist.created_at,
            now,
        )
        return entry

    async def remove(
        self,
        area_id: str,
        playlist_id: UUID,
        entry_id: UUID,
    ) -> PlaylistTrackRemoval:
        playlist = await self.get(area_id, playlist_id)
        if playlist is None:
            raise MusicPlaylistNotFoundError
        retained = tuple(entry for entry in playlist.entries if entry.id != entry_id)
        if len(retained) == len(playlist.entries):
            return PlaylistTrackRemoval(playlist_id, entry_id, False)
        now = datetime.now(UTC)
        compacted = tuple(
            MusicPlaylistEntry(
                entry.id,
                entry.playlist_id,
                position,
                entry.track,
                entry.added_by_person_id,
                entry.created_at,
            )
            for position, entry in enumerate(retained, start=1)
        )
        self.playlists[playlist_id] = MusicPlaylist(
            playlist.id,
            playlist.area_id,
            playlist.name,
            playlist.normalized_name,
            playlist.created_by_person_id,
            compacted,
            playlist.created_at,
            now,
        )
        return PlaylistTrackRemoval(playlist_id, entry_id, True)

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
        if len(tracks) > max_tracks:
            raise MusicPlaylistFullError
        playlist = await self.create(
            area_id,
            name,
            normalized_name,
            created_by_person_id,
        )
        for track in tracks:
            await self.append(
                area_id,
                playlist.id,
                track,
                created_by_person_id,
                max_tracks=max_tracks,
            )
        imported = await self.get(area_id, playlist.id)
        assert imported is not None
        return imported

    async def rename(
        self,
        area_id: str,
        playlist_id: UUID,
        name: str,
        normalized_name: str,
    ) -> PlaylistRename:
        playlist = await self.get(area_id, playlist_id)
        if playlist is None:
            raise MusicPlaylistNotFoundError
        if any(
            item.id != playlist_id
            and item.area_id == area_id
            and item.normalized_name == normalized_name
            for item in self.playlists.values()
        ):
            raise MusicPlaylistConflictError
        changed = playlist.name != name or playlist.normalized_name != normalized_name
        if changed:
            self.playlists[playlist_id] = MusicPlaylist(
                playlist.id,
                playlist.area_id,
                name,
                normalized_name,
                playlist.created_by_person_id,
                playlist.entries,
                playlist.created_at,
                datetime.now(UTC),
            )
        return PlaylistRename(playlist_id, playlist.name, name, changed)

    async def delete(self, area_id: str, playlist_id: UUID) -> PlaylistDeletion:
        playlist = await self.get(area_id, playlist_id)
        if playlist is None:
            return PlaylistDeletion(playlist_id, None, False, 0)
        del self.playlists[playlist_id]
        return PlaylistDeletion(playlist_id, playlist.name, True, playlist.track_count)

    async def clear(self, area_id: str, playlist_id: UUID) -> PlaylistClear:
        playlist = await self.get(area_id, playlist_id)
        if playlist is None:
            raise MusicPlaylistNotFoundError
        self.playlists[playlist_id] = MusicPlaylist(
            playlist.id,
            playlist.area_id,
            playlist.name,
            playlist.normalized_name,
            playlist.created_by_person_id,
            (),
            playlist.created_at,
            datetime.now(UTC),
        )
        return PlaylistClear(playlist_id, playlist.name, playlist.track_count)


@dataclass
class FakeMusic:
    replaced: tuple[MusicTrack, ...] = ()

    async def search(self, query: str, *, source=None, limit: int | None = None):
        assert limit == 1
        return (MusicTrack(source or "netease", query, query, ("artist",), 1000),)

    async def lookup(self, reference):
        return MusicTrack(
            reference.source,
            reference.source_id,
            reference.source_id,
            ("artist",),
            1000,
        )

    async def track_from_input(self, value: str, *, source=None):
        return MusicTrack(source or "netease", value, value, ("artist",), 1000)

    async def replace_queue(
        self,
        caller: AgentIdentity,
        tracks: tuple[MusicTrack, ...],
    ) -> QueueRebuildResult:
        self.replaced = tracks
        return QueueRebuildResult(
            VoiceChannelKey(caller.conversation.area_id, "voice"),
            len(tracks),
            True,
            False,
        )


@dataclass
class FakePlaylistSource:
    snapshot: NeteasePlaylistSnapshot
    references: list[tuple[str, int]]

    async def playlist(
        self,
        reference: str,
        *,
        limit: int,
    ) -> NeteasePlaylistSnapshot:
        self.references.append((reference, limit))
        return self.snapshot


def service() -> tuple[MusicPlaylistService, InMemoryPlaylistRepository, FakeMusic]:
    repository = InMemoryPlaylistRepository()
    music = FakeMusic()
    return (
        MusicPlaylistService(
            settings(),
            repository,
            music,  # type: ignore[arg-type]
        ),
        repository,
        music,
    )


@pytest.mark.asyncio
async def test_playlists_are_shared_per_area_and_rebuild_the_queue() -> None:
    playlists, _, music = service()
    created = await playlists.create(identity(), "  夜间   电台  ")
    same_area_member = identity("other")

    assert created.name == "夜间 电台"
    assert (await playlists.list(same_area_member))[0].id == created.id
    with pytest.raises(MusicPlaylistConflictError):
        await playlists.create(same_area_member, "夜间 电台")

    first = await playlists.add(identity(), created.id, "first")
    second = await playlists.add(same_area_member, created.id, "second")
    loaded = await playlists.get(identity(), created.id)
    assert [entry.position for entry in loaded.entries] == [1, 2]
    assert [entry.track.title for entry in loaded.entries] == ["first", "second"]

    removed = await playlists.remove(same_area_member, created.id, first.id)
    assert removed.removed is True
    compacted = await playlists.get(identity(), created.id)
    assert [(entry.position, entry.id) for entry in compacted.entries] == [(1, second.id)]

    result = await playlists.load(identity(), created.id)
    assert result.queue.loaded_count == 1
    assert [track.title for track in music.replaced] == ["second"]

    with pytest.raises(MusicPlaylistNotFoundError):
        await playlists.get(identity(area_id="other-area"), created.id)


@pytest.mark.asyncio
async def test_playlist_service_persists_and_rebuilds_mixed_source_references() -> None:
    playlists, _, music = service()
    created = await playlists.create(identity(), "跨平台歌单")

    youtube = await playlists.add_reference(
        identity(),
        created.id,
        MusicTrackReference(MusicSourceKind.YOUTUBE, "dQw4w9WgXcQ"),
    )
    bilibili = await playlists.add_reference(
        identity(),
        created.id,
        MusicTrackReference(MusicSourceKind.BILIBILI, "BV13x41117TL:p=1"),
    )
    loaded = await playlists.load(identity(), created.id)

    assert youtube.track.reference == MusicTrackReference(
        MusicSourceKind.YOUTUBE,
        "dQw4w9WgXcQ",
    )
    assert bilibili.track.reference == MusicTrackReference(
        MusicSourceKind.BILIBILI,
        "BV13x41117TL:p=1",
    )
    assert loaded.queue.loaded_count == 2
    assert [track.source for track in music.replaced] == [
        MusicSourceKind.YOUTUBE,
        MusicSourceKind.BILIBILI,
    ]


@pytest.mark.asyncio
async def test_playlist_service_rejects_private_scope_and_empty_load() -> None:
    playlists, _, _ = service()
    with pytest.raises(MusicAreaRequiredError):
        await playlists.list(identity(area_id=""))

    empty = await playlists.create(identity(), "empty")
    with pytest.raises(MusicPlaylistEmptyError):
        await playlists.load(identity(), empty.id)


@pytest.mark.asyncio
async def test_playlist_service_renames_clears_and_deletes_inside_one_area() -> None:
    playlists, _, _ = service()
    first = await playlists.create(identity(), " First ")
    second = await playlists.create(identity(), "Second")
    await playlists.add(identity(), first.id, "Melt")

    renamed = await playlists.rename(identity("other"), first.id, "  Favorites  ")
    unchanged = await playlists.rename(identity(), first.id, "Favorites")
    assert (renamed.old_name, renamed.new_name, renamed.changed) == (
        "First",
        "Favorites",
        True,
    )
    assert unchanged.changed is False
    with pytest.raises(MusicPlaylistConflictError):
        await playlists.rename(identity(), first.id, second.name.casefold())

    cleared = await playlists.clear(identity("other"), first.id)
    cleared_again = await playlists.clear(identity(), first.id)
    assert cleared.removed_track_count == 1
    assert cleared_again.removed_track_count == 0
    assert (await playlists.get(identity(), first.id)).entries == ()

    deleted = await playlists.delete(identity(), first.id)
    deleted_again = await playlists.delete(identity(), first.id)
    assert (deleted.name, deleted.deleted, deleted.removed_track_count) == (
        "Favorites",
        True,
        0,
    )
    assert deleted_again == PlaylistDeletion(first.id, None, False, 0)
    assert (await playlists.delete(identity(area_id="other-area"), second.id)).deleted is False


def tool_context() -> ToolExecutionContext:
    caller = identity()
    return ToolExecutionContext(
        uuid4(),
        caller,
        AgentRunLimits(),
        (
            "create_music_playlist",
            "list_music_playlists",
            "get_music_playlist",
            "add_music_playlist_track",
            "remove_music_playlist_track",
            "rename_music_playlist",
            "delete_music_playlist",
            "clear_music_playlist",
            "load_music_playlist",
            "preview_netease_playlist",
            "import_netease_playlist",
        ),
    )


@pytest.mark.asyncio
async def test_playlist_agent_tools_cover_the_shared_playlist_lifecycle() -> None:
    playlists, _, music = service()
    options = {"timeout_seconds": 1, "max_output_characters": 4000}
    create = CreateMusicPlaylistTool(playlists, **options)
    list_all = ListMusicPlaylistsTool(playlists, **options)
    get = GetMusicPlaylistTool(playlists, **options)
    add = AddMusicPlaylistTrackTool(playlists, **options)
    remove = RemoveMusicPlaylistTrackTool(playlists, **options)
    rename = RenameMusicPlaylistTool(playlists, **options)
    clear = ClearMusicPlaylistTool(playlists, **options)
    delete = DeleteMusicPlaylistTool(playlists, **options)
    load = LoadMusicPlaylistTool(playlists, **options)
    context = tool_context()

    created = await create.execute(context, CreateMusicPlaylistInput(name="favorites"))
    playlist_id = created.playlist.id
    added = await add.execute(
        context,
        AddMusicPlaylistTrackInput(playlist_id=playlist_id, query="Melt"),
    )
    listed = await list_all.execute(context, EmptyToolInput())
    detailed = await get.execute(context, PlaylistIdInput(playlist_id=playlist_id))
    loaded = await load.execute(context, PlaylistIdInput(playlist_id=playlist_id))
    removed = await remove.execute(
        context,
        RemoveMusicPlaylistTrackInput(
            playlist_id=playlist_id,
            entry_id=added.entry.id,
        ),
    )
    renamed = await rename.execute(
        context,
        RenameMusicPlaylistInput(playlist_id=playlist_id, name="Miku Favorites"),
    )
    await add.execute(
        context,
        AddMusicPlaylistTrackInput(playlist_id=playlist_id, query="World is Mine"),
    )
    cleared = await clear.execute(context, PlaylistIdInput(playlist_id=playlist_id))
    deleted = await delete.execute(context, PlaylistIdInput(playlist_id=playlist_id))
    deleted_again = await delete.execute(context, PlaylistIdInput(playlist_id=playlist_id))

    assert listed.playlists[0].track_count == 1
    assert detailed.entries[0].track.title == "Melt"
    assert loaded.loaded_count == 1
    assert loaded.replaced_current is True
    assert [track.title for track in music.replaced] == ["Melt"]
    assert removed.removed is True
    assert renamed.new_name == "Miku Favorites"
    assert renamed.changed is True
    assert cleared.removed_track_count == 1
    assert deleted.name == "Miku Favorites"
    assert deleted.deleted is True
    assert deleted_again.deleted is False


@pytest.mark.asyncio
async def test_playlist_tool_maps_missing_area_to_stable_error() -> None:
    playlists, _, _ = service()
    tool = ListMusicPlaylistsTool(
        playlists,
        timeout_seconds=1,
        max_output_characters=1000,
    )
    context = tool_context()
    private_context = ToolExecutionContext(
        context.run_id,
        identity(area_id=""),
        context.limits,
        context.enabled_tools,
    )

    with pytest.raises(ToolExecutionError) as error:
        await tool.execute(private_context, EmptyToolInput())

    assert error.value.error_code == "music_area_required"


def import_service(
    snapshot: NeteasePlaylistSnapshot,
) -> tuple[MusicPlaylistService, InMemoryPlaylistRepository, FakePlaylistSource]:
    repository = InMemoryPlaylistRepository()
    music = FakeMusic()
    source = FakePlaylistSource(snapshot, [])
    return (
        MusicPlaylistService(
            settings(),
            repository,
            music,  # type: ignore[arg-type]
            source,
        ),
        repository,
        source,
    )


def source_snapshot(
    *,
    declared: int = 2,
    tracks: tuple[MusicTrack, ...] | None = None,
) -> NeteasePlaylistSnapshot:
    values = tracks or (
        MusicTrack("netease", "39", "39", ("初音未来",), 222000),
        MusicTrack("netease", "831", "Tell Your World", ("初音未来",), 245000),
    )
    return NeteasePlaylistSnapshot("24381616", "Miku Favorites", declared, values)


@pytest.mark.asyncio
async def test_netease_playlist_import_is_ordered_area_scoped_and_atomic() -> None:
    playlists, repository, source = import_service(source_snapshot())

    preview = await playlists.preview_netease("24381616")
    imported = await playlists.import_netease(
        identity(),
        "24381616",
        name="未来歌单",
        allow_partial=False,
    )

    assert preview.complete is True
    assert source.references == [("24381616", 3), ("24381616", 3)]
    assert imported.playlist.name == "未来歌单"
    assert imported.imported_track_count == 2
    assert imported.skipped_track_count == 0
    assert [entry.track.source_id for entry in imported.playlist.entries] == ["39", "831"]
    assert (await repository.list("area"))[0].id == imported.playlist.id
    assert await repository.get("other-area", imported.playlist.id) is None


@pytest.mark.asyncio
async def test_netease_playlist_import_requires_explicit_partial_consent() -> None:
    incomplete = source_snapshot(
        declared=4,
        tracks=(
            MusicTrack("netease", "39", "39", ("初音未来",), 222000),
            MusicTrack("netease", "831", "Tell Your World", ("初音未来",), 245000),
            MusicTrack("netease", "123", "Hand in Hand", ("初音未来",), 250000),
        ),
    )
    playlists, repository, _ = import_service(incomplete)

    with pytest.raises(NeteasePlaylistTooLargeError):
        await playlists.import_netease(
            identity(),
            "24381616",
            name=None,
            allow_partial=False,
        )
    assert await repository.list("area") == ()

    imported = await playlists.import_netease(
        identity(),
        "24381616",
        name=None,
        allow_partial=True,
    )
    assert imported.partial is True
    assert imported.imported_track_count == 3
    assert imported.skipped_track_count == 1

    missing_track = source_snapshot(declared=3)
    incomplete_service, _, _ = import_service(missing_track)
    with pytest.raises(NeteasePlaylistIncompleteError):
        await incomplete_service.import_netease(
            identity(),
            "other",
            name="Incomplete",
            allow_partial=False,
        )


@pytest.mark.asyncio
async def test_netease_playlist_tools_preview_then_import_without_exposing_all_tracks() -> None:
    playlists, _, _ = import_service(source_snapshot())
    options = {"timeout_seconds": 1, "max_output_characters": 4000}
    preview_tool = PreviewNeteasePlaylistTool(playlists, **options)
    import_tool = ImportNeteasePlaylistTool(playlists, **options)

    preview = await preview_tool.execute(
        tool_context(),
        NeteasePlaylistReferenceInput(reference="24381616"),
    )
    imported = await import_tool.execute(
        tool_context(),
        ImportNeteasePlaylistInput(
            reference="24381616",
            name="Imported",
        ),
    )

    assert preview.name == "Miku Favorites"
    assert preview.visible_track_count == 2
    assert preview.complete is True
    assert [track.source_id for track in preview.preview_tracks] == ["39", "831"]
    assert imported.playlist.name == "Imported"
    assert imported.imported_track_count == 2
    assert imported.partial is False


@pytest.mark.asyncio
async def test_netease_playlist_tool_maps_invalid_reference_to_stable_error() -> None:
    class InvalidSource:
        async def playlist(self, reference: str, *, limit: int) -> NeteasePlaylistSnapshot:
            del reference, limit
            raise NeteasePlaylistReferenceError

    repository = InMemoryPlaylistRepository()
    playlists = MusicPlaylistService(
        settings(),
        repository,
        FakeMusic(),
        InvalidSource(),
    )
    tool = PreviewNeteasePlaylistTool(
        playlists,
        timeout_seconds=1,
        max_output_characters=4000,
    )

    with pytest.raises(ToolExecutionError) as error:
        await tool.execute(
            tool_context(),
            NeteasePlaylistReferenceInput(reference="invalid"),
        )

    assert error.value.error_code == "invalid_netease_playlist_reference"
