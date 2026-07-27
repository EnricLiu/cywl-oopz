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
    CreateMusicPlaylistInput,
    CreateMusicPlaylistTool,
    GetMusicPlaylistTool,
    ListMusicPlaylistsTool,
    LoadMusicPlaylistTool,
    PlaylistIdInput,
    RemoveMusicPlaylistTrackInput,
    RemoveMusicPlaylistTrackTool,
)
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.music.errors import (
    MusicAreaRequiredError,
    MusicPlaylistConflictError,
    MusicPlaylistEmptyError,
    MusicPlaylistFullError,
    MusicPlaylistNotFoundError,
)
from cywl_oopz.features.music.models import (
    MusicPlaylist,
    MusicPlaylistEntry,
    MusicPlaylistSummary,
    MusicTrack,
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


@dataclass
class FakeMusic:
    replaced: tuple[MusicTrack, ...] = ()

    async def search(self, query: str, *, limit: int | None = None):
        assert limit == 1
        return (MusicTrack("netease", query, query, ("artist",), 1000),)

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
async def test_playlist_service_rejects_private_scope_and_empty_load() -> None:
    playlists, _, _ = service()
    with pytest.raises(MusicAreaRequiredError):
        await playlists.list(identity(area_id=""))

    empty = await playlists.create(identity(), "empty")
    with pytest.raises(MusicPlaylistEmptyError):
        await playlists.load(identity(), empty.id)


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
            "load_music_playlist",
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

    assert listed.playlists[0].track_count == 1
    assert detailed.entries[0].track.title == "Melt"
    assert loaded.loaded_count == 1
    assert loaded.replaced_current is True
    assert [track.title for track in music.replaced] == ["Melt"]
    assert removed.removed is True


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
