from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.features.agent.models import AgentIdentity
from cywl_oopz.features.music.commands import MusicCommand, MusicCommandRenderer
from cywl_oopz.features.music.errors import (
    MusicPlaylistConflictError,
    MusicVoiceChannelRequiredError,
)
from cywl_oopz.features.music.models import (
    EnqueueResult,
    MusicPlaybackPolicy,
    MusicPlaylist,
    MusicPlaylistEntry,
    MusicPlaylistSummary,
    MusicProviderHealth,
    MusicProviderHealthState,
    MusicQueueClearResult,
    MusicQueueSnapshot,
    MusicSourceKind,
    MusicTrack,
    NeteasePlaylistImport,
    NeteasePlaylistSnapshot,
    PlaybackOrder,
    PlaybackPolicyChange,
    PlaybackState,
    PlaylistClear,
    PlaylistDeletion,
    PlaylistQueueLoad,
    PlaylistRename,
    PlaylistTrackRemoval,
    QueuedTrack,
    QueueRebuildResult,
    RepeatPolicy,
    VoiceChannelKey,
)
from cywl_oopz.testing.commands import dispatch_command


@dataclass
class FakeMessage:
    plain_text: str
    sender_id: str = "person"
    area: str = "area"
    channel: str = "text"
    message_id: str = "message"
    text: str = ""
    content: str = ""


@dataclass
class FakeContext:
    event: object
    replies: list[str] = field(default_factory=list)

    async def reply(self, text: str) -> None:
        self.replies.append(text)


def context(message: FakeMessage, *, private: bool = False) -> FakeContext:
    return FakeContext(SimpleNamespace(message=message, is_private=private))


def track(source_id: str, title: str | None = None) -> MusicTrack:
    return MusicTrack("netease", source_id, title or source_id, ("初音未来",), 180_000)


class StubMusic:
    def __init__(self) -> None:
        self.channel = VoiceChannelKey("area", "voice")
        self.current = QueuedTrack(track("39", "39"), "person")
        self.upcoming = (QueuedTrack(track("melt", "Melt"), "person"),)
        self.policy = MusicPlaybackPolicy()
        self.calls: list[tuple[object, ...]] = []
        self.default_source = MusicSourceKind.NETEASE

    async def search(self, query: str, *, source=None, limit: int | None = None):
        self.calls.append(("search", query, source, limit))
        target_source = source or MusicSourceKind.NETEASE
        return (
            MusicTrack(target_source, "39", "39", ("初音未来",), 180_000),
            MusicTrack(target_source, "melt", "Melt", ("初音未来",), 180_000),
        )[:limit]

    async def enqueue_input(
        self,
        identity: AgentIdentity,
        value: str,
        *,
        source=None,
        idempotency_key: str = "",
    ) -> EnqueueResult:
        self.calls.append(("enqueue_input", identity, value, source, idempotency_key))
        item = QueuedTrack(
            MusicTrack(source or "netease", value, value, ("初音未来",), 180_000),
            identity.person_id,
        )
        return EnqueueResult(self.channel, item, 2, False)

    async def health(self) -> tuple[MusicProviderHealth, ...]:
        self.calls.append(("health",))
        return (
            MusicProviderHealth(
                MusicSourceKind.NETEASE,
                MusicProviderHealthState.READY,
            ),
            MusicProviderHealth(
                MusicSourceKind.BILIBILI,
                MusicProviderHealthState.DEGRADED,
                "anonymous search limited",
            ),
        )

    async def queue(self, identity: AgentIdentity) -> MusicQueueSnapshot:
        self.calls.append(("queue", identity))
        return MusicQueueSnapshot(
            self.channel,
            PlaybackState.PLAYING,
            self.policy,
            self.current,
            self.upcoming,
            0,
            3,
        )

    async def set_policy(
        self,
        identity: AgentIdentity,
        *,
        order: PlaybackOrder | None = None,
        repeat: RepeatPolicy | None = None,
    ) -> PlaybackPolicyChange:
        self.calls.append(("set_policy", identity, order, repeat))
        policy = MusicPlaybackPolicy(order or self.policy.order, repeat or self.policy.repeat)
        changed = policy != self.policy
        self.policy = policy
        return PlaybackPolicyChange(self.channel, policy, changed)

    async def skip(self, identity: AgentIdentity) -> bool:
        self.calls.append(("skip", identity))
        return True

    async def pause(self, identity: AgentIdentity) -> bool:
        self.calls.append(("pause", identity))
        return True

    async def resume(self, identity: AgentIdentity) -> bool:
        self.calls.append(("resume", identity))
        return True

    async def clear(self, identity: AgentIdentity) -> MusicQueueClearResult:
        self.calls.append(("clear", identity))
        return MusicQueueClearResult(self.channel, True, 3)


class StubPlaylists:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.first_id = UUID("00000000-0000-0000-0000-000000000001")
        self.second_id = UUID("00000000-0000-0000-0000-000000000002")
        self.entry_id = UUID("10000000-0000-0000-0000-000000000001")
        self.first = MusicPlaylist(
            self.first_id,
            "area",
            "未来歌单",
            "未来歌单",
            "person",
            (
                MusicPlaylistEntry(
                    self.entry_id,
                    self.first_id,
                    1,
                    track("39", "39"),
                    "person",
                    now,
                ),
            ),
            now,
            now,
        )
        self.second = MusicPlaylist(
            self.second_id,
            "area",
            "夜间电台",
            "夜间电台",
            "person",
            (),
            now,
            now,
        )
        self.calls: list[tuple[object, ...]] = []

    async def list(self, identity: AgentIdentity):
        self.calls.append(("list", identity))
        return tuple(
            MusicPlaylistSummary(
                playlist.id,
                playlist.area_id,
                playlist.name,
                playlist.track_count,
                playlist.updated_at,
            )
            for playlist in (self.first, self.second)
        )

    async def get(self, identity: AgentIdentity, playlist_id: UUID) -> MusicPlaylist:
        self.calls.append(("get", identity, playlist_id))
        return self.first if playlist_id == self.first_id else self.second

    async def create(self, identity: AgentIdentity, name: str) -> MusicPlaylist:
        self.calls.append(("create", identity, name))
        return self.second

    async def rename(
        self,
        identity: AgentIdentity,
        playlist_id: UUID,
        name: str,
    ) -> PlaylistRename:
        self.calls.append(("rename", identity, playlist_id, name))
        return PlaylistRename(playlist_id, self.first.name, name, True)

    async def delete(self, identity: AgentIdentity, playlist_id: UUID) -> PlaylistDeletion:
        self.calls.append(("delete", identity, playlist_id))
        return PlaylistDeletion(playlist_id, self.first.name, True, 1)

    async def clear(self, identity: AgentIdentity, playlist_id: UUID) -> PlaylistClear:
        self.calls.append(("clear", identity, playlist_id))
        return PlaylistClear(playlist_id, self.first.name, 1)

    async def add_input(
        self,
        identity: AgentIdentity,
        playlist_id: UUID,
        value: str,
        *,
        source=None,
    ) -> MusicPlaylistEntry:
        self.calls.append(("add_input", identity, playlist_id, value, source))
        entry = self.first.entries[0]
        if source is None:
            return entry
        return MusicPlaylistEntry(
            entry.id,
            entry.playlist_id,
            entry.position,
            MusicTrack(source, "39", "39", ("初音未来",), 180_000),
            entry.added_by_person_id,
            entry.created_at,
        )

    async def remove(
        self,
        identity: AgentIdentity,
        playlist_id: UUID,
        entry_id: UUID,
    ) -> PlaylistTrackRemoval:
        self.calls.append(("remove", identity, playlist_id, entry_id))
        return PlaylistTrackRemoval(playlist_id, entry_id, True)

    async def load(self, identity: AgentIdentity, playlist_id: UUID) -> PlaylistQueueLoad:
        self.calls.append(("load", identity, playlist_id))
        return PlaylistQueueLoad(
            playlist_id,
            self.first.name,
            QueueRebuildResult(VoiceChannelKey("area", "voice"), 1, True, False),
        )

    async def preview_netease(self, reference: str) -> NeteasePlaylistSnapshot:
        self.calls.append(("preview", reference))
        return NeteasePlaylistSnapshot("24381616", "云端歌单", 2, (track("39", "39"),))

    async def import_netease(
        self,
        identity: AgentIdentity,
        reference: str,
        *,
        name: str | None,
        allow_partial: bool,
    ) -> NeteasePlaylistImport:
        self.calls.append(("import", identity, reference, name, allow_partial))
        return NeteasePlaylistImport("24381616", "云端歌单", 2, self.first)


def fixture() -> tuple[CommandRouter, StubMusic, StubPlaylists]:
    music = StubMusic()
    playlists = StubPlaylists()
    router = CommandRouter("!")
    router.register_definition(
        MusicCommand(music, playlists, "!").definition()  # type: ignore[arg-type]
    )
    return router, music, playlists


async def dispatch(router: CommandRouter, text: str) -> FakeContext:
    message = FakeMessage(text)
    target = context(message)
    assert await dispatch_command(router, message, target) is True
    return target


@pytest.mark.asyncio
async def test_music_command_playback_search_queue_policy_and_controls() -> None:
    router, music, _ = fixture()

    help_result = await dispatch(router, "!music")
    played = await dispatch(router, "!music play Tell Your World")
    searched = await dispatch(router, "!music search 初音未来")
    queued = await dispatch(router, "!music queue")
    mode = await dispatch(router, "!music mode shuffle all")
    current_mode = await dispatch(router, "!music mode")
    skipped = await dispatch(router, "!music skip")
    paused = await dispatch(router, "!music pause")
    resumed = await dispatch(router, "!music resume")
    cleared = await dispatch(router, "!music clear")

    assert "Music 命令" in help_result.replies[0]
    assert "Tell Your World" in played.replies[0]
    assert "找到 2 首" in searched.replies[0]
    assert "正在播放" in queued.replies[0]
    assert "随机 · 列表循环" in mode.replies[0]
    assert "当前播放策略" in current_mode.replies[0]
    assert "切换下一首" in skipped.replies[0]
    assert "已暂停" in paused.replies[0]
    assert "继续播放" in resumed.replies[0]
    assert "移除 3 首" in cleared.replies[0]
    enqueue = next(call for call in music.calls if call[0] == "enqueue_input")
    assert enqueue[2:] == ("Tell Your World", None, "music-command:message")


@pytest.mark.asyncio
async def test_music_command_routes_sources_urls_and_health_explicitly() -> None:
    router, music, playlists = fixture()

    played = await dispatch(
        router,
        "!music play --source youtube https://youtu.be/dQw4w9WgXcQ",
    )
    searched = await dispatch(router, "!music search --source bilibili 初音未来")
    added = await dispatch(
        router,
        "!music playlist add #1 --source youtube Tell Your World",
    )
    sources = await dispatch(router, "!music sources")
    misplaced = await dispatch(router, "!music search 初音未来 --source youtube")

    enqueue = next(call for call in music.calls if call[0] == "enqueue_input")
    search = next(call for call in music.calls if call[0] == "search")
    playlist_add = next(call for call in playlists.calls if call[0] == "add_input")
    assert enqueue[2:4] == (
        "https://youtu.be/dQw4w9WgXcQ",
        MusicSourceKind.YOUTUBE,
    )
    assert search[1:] == ("初音未来", MusicSourceKind.BILIBILI, 5)
    assert playlist_add[2:] == (
        playlists.first_id,
        "Tell Your World",
        MusicSourceKind.YOUTUBE,
    )
    assert "▶️" in played.replies[0]
    assert "Bilibili · 找到 2 首" in searched.replies[0]
    assert "▶️" in added.replies[0]
    assert "✅ 网易云 · 默认" in sources.replies[0]
    assert "⚠️ Bilibili · anonymous search limited" in sources.replies[0]
    assert "Music 命令" in misplaced.replies[0]


@pytest.mark.asyncio
async def test_music_command_manages_playlists_with_human_friendly_indexes() -> None:
    router, _, playlists = fixture()

    listed = await dispatch(router, "!music playlist list")
    shown = await dispatch(router, "!music playlist show #1")
    created = await dispatch(router, "!music playlist create 深夜 电台")
    renamed = await dispatch(router, "!music playlist rename #1 初音收藏")
    added = await dispatch(router, "!music playlist add #1 Tell Your World")
    removed = await dispatch(router, "!music playlist remove #1 1")
    loaded = await dispatch(router, "!music playlist load #1")
    cleared = await dispatch(router, "!music playlist clear #1")
    deleted = await dispatch(router, "!music playlist delete #1")

    assert "#1 **未来歌单**" in listed.replies[0]
    assert "**未来歌单** · 1 首" in shown.replies[0]
    assert "夜间电台" in created.replies[0]
    assert "初音收藏" in renamed.replies[0]
    assert "已把 ☁️ **39**" in added.replies[0]
    assert "移除歌曲" in removed.replies[0]
    assert "重建播放队列" in loaded.replies[0]
    assert "移除 1 首" in cleared.replies[0]
    assert "已删除" in deleted.replies[0]
    rename_call = next(call for call in playlists.calls if call[0] == "rename")
    assert rename_call[2:] == (playlists.first_id, "初音收藏")
    remove_call = next(call for call in playlists.calls if call[0] == "remove")
    assert remove_call[2:] == (playlists.first_id, playlists.entry_id)
    assert sum(call[0] == "load" for call in playlists.calls) == 1


@pytest.mark.asyncio
async def test_music_command_previews_and_explicitly_imports_partial_netease_playlist() -> None:
    router, _, playlists = fixture()

    preview = await dispatch(router, "!music playlist preview 24381616")
    imported = await dispatch(
        router,
        "!music playlist import 24381616 --partial 新的 云端 歌单",
    )

    assert "可读取：1/2 首" in preview.replies[0]
    assert "需要 --partial" in preview.replies[0]
    assert "已导入为 **未来歌单**" in imported.replies[0]
    import_call = next(call for call in playlists.calls if call[0] == "import")
    assert import_call[2:] == ("24381616", "新的 云端 歌单", True)


@pytest.mark.asyncio
async def test_music_command_maps_expected_errors_and_invalid_syntax() -> None:
    class MissingVoiceMusic(StubMusic):
        async def enqueue_input(
            self,
            identity,
            value,
            *,
            source=None,
            idempotency_key="",
        ):
            del identity, value, source, idempotency_key
            raise MusicVoiceChannelRequiredError

    class ConflictingPlaylists(StubPlaylists):
        async def create(self, identity, name):
            del identity, name
            raise MusicPlaylistConflictError

    router = CommandRouter("!")
    router.register_definition(
        MusicCommand(  # type: ignore[arg-type]
            MissingVoiceMusic(),
            ConflictingPlaylists(),
            "!",
        ).definition()
    )

    missing_voice = await dispatch(router, "!music play 39")
    conflict = await dispatch(router, "!music playlist create favorites")
    invalid = await dispatch(router, "!music mode impossible")

    assert missing_voice.replies == ["请先加入当前 area 的语音频道。"]
    assert conflict.replies == ["当前 area 已有同名共享歌单。"]
    assert "Music 命令" in invalid.replies[0]


def test_music_command_renderer_bounds_large_queue_and_playlist_outputs() -> None:
    renderer = MusicCommandRenderer("!")
    channel = VoiceChannelKey("area", "voice")
    queue = MusicQueueSnapshot(
        channel,
        PlaybackState.PLAYING,
        MusicPlaybackPolicy(),
        QueuedTrack(track("current"), "person"),
        tuple(
            QueuedTrack(
                MusicTrack(
                    "netease",
                    str(index),
                    "很长的歌曲标题" * 100,
                    ("很长的歌手名称" * 100,),
                ),
                "person",
            )
            for index in range(40)
        ),
        0,
        1,
    )
    now = datetime.now(UTC)
    playlist = MusicPlaylist(
        uuid4(),
        "area",
        "大型歌单",
        "大型歌单",
        "person",
        tuple(
            MusicPlaylistEntry(
                uuid4(),
                uuid4(),
                index,
                track(str(index)),
                "person",
                now,
            )
            for index in range(1, 50)
        ),
        now,
        now,
    )

    queue_text = renderer.queue(queue)
    playlist_text = renderer.playlist(playlist)

    assert "还有 32 首" in queue_text
    assert "还有 37 首" in playlist_text
    assert "☁️" in queue_text
    assert "☁️" in playlist_text
    assert len(queue_text) < 2000
    assert len(playlist_text) < 2000
