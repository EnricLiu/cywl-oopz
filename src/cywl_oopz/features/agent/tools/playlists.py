"""Agent tool adapters for area-shared music playlists."""

from __future__ import annotations

from typing import Never
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.features.music.errors import (
    MusicAreaRequiredError,
    MusicCatalogError,
    MusicError,
    MusicNotFoundError,
    MusicPlaybackError,
    MusicPlaylistConflictError,
    MusicPlaylistEmptyError,
    MusicPlaylistFullError,
    MusicPlaylistNameError,
    MusicPlaylistNotFoundError,
    MusicQueryError,
    MusicQueueFullError,
    MusicVoiceChannelRequiredError,
)
from cywl_oopz.features.music.models import (
    MusicPlaylist,
    MusicPlaylistEntry,
    MusicPlaylistSummary,
)
from cywl_oopz.features.music.playlists import MusicPlaylistService

from .builtin import EmptyToolInput
from .models import (
    ToolDescriptor,
    ToolEffect,
    ToolExecutionContext,
    ToolExecutionError,
)
from .music import MusicTrackOutput


class _PlaylistTool:
    """Map playlist and queue failures to stable Agent error codes."""

    _ERROR_CODES = {
        MusicAreaRequiredError: "music_area_required",
        MusicPlaylistNameError: "invalid_music_playlist_name",
        MusicPlaylistConflictError: "music_playlist_exists",
        MusicPlaylistNotFoundError: "music_playlist_not_found",
        MusicPlaylistFullError: "music_playlist_full",
        MusicPlaylistEmptyError: "music_playlist_empty",
        MusicVoiceChannelRequiredError: "music_voice_channel_required",
        MusicQueryError: "invalid_music_query",
        MusicQueueFullError: "music_queue_full",
        MusicNotFoundError: "music_not_found",
        MusicCatalogError: "music_catalog_unavailable",
        MusicPlaybackError: "music_playback_failed",
    }

    @classmethod
    def _raise_tool_error(cls, error: Exception) -> Never:
        if isinstance(error, DatabaseError):
            raise ToolExecutionError("music_playlist_unavailable") from error
        for error_type, code in cls._ERROR_CODES.items():
            if isinstance(error, error_type):
                raise ToolExecutionError(code) from error
        raise ToolExecutionError("music_playlist_failed") from error


class PlaylistSummaryOutput(BaseModel):
    """Compact shared playlist metadata."""

    id: UUID
    name: str
    track_count: int

    @classmethod
    def from_summary(cls, playlist: MusicPlaylistSummary) -> PlaylistSummaryOutput:
        return cls(
            id=playlist.id,
            name=playlist.name,
            track_count=playlist.track_count,
        )

    @classmethod
    def from_playlist(cls, playlist: MusicPlaylist) -> PlaylistSummaryOutput:
        return cls(
            id=playlist.id,
            name=playlist.name,
            track_count=playlist.track_count,
        )


class PlaylistEntryOutput(BaseModel):
    """Stable entry identity, order, and catalog metadata."""

    id: UUID
    position: int
    track: MusicTrackOutput

    @classmethod
    def from_entry(cls, entry: MusicPlaylistEntry) -> PlaylistEntryOutput:
        return cls(
            id=entry.id,
            position=entry.position,
            track=MusicTrackOutput.from_track(entry.track),
        )


class MusicPlaylistOutput(BaseModel):
    """One complete bounded playlist."""

    id: UUID
    name: str
    entries: tuple[PlaylistEntryOutput, ...]

    @classmethod
    def from_playlist(cls, playlist: MusicPlaylist) -> MusicPlaylistOutput:
        return cls(
            id=playlist.id,
            name=playlist.name,
            entries=tuple(PlaylistEntryOutput.from_entry(entry) for entry in playlist.entries),
        )


class CreateMusicPlaylistInput(BaseModel):
    """Name of a new area-shared playlist."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80, description="歌单名称")


class CreateMusicPlaylistOutput(BaseModel):
    """Created playlist metadata."""

    playlist: PlaylistSummaryOutput


class CreateMusicPlaylistTool(_PlaylistTool):
    """Create an empty playlist shared by the current OOPZ area."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="create_music_playlist",
            display_name="新建歌单",
            description="在当前 OOPZ area 新建一个所有成员共享的空歌单。",
            input_model=CreateMusicPlaylistInput,
            output_model=CreateMusicPlaylistOutput,
            effect=ToolEffect.WRITE,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=False,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        values = CreateMusicPlaylistInput.model_validate(arguments)
        try:
            playlist = await self._playlists.create(context.identity, values.name)
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return CreateMusicPlaylistOutput(playlist=PlaylistSummaryOutput.from_playlist(playlist))


class ListMusicPlaylistsOutput(BaseModel):
    """All shared playlists in the current area."""

    playlists: tuple[PlaylistSummaryOutput, ...]


class ListMusicPlaylistsTool(_PlaylistTool):
    """Discover playlist IDs before read or mutation calls."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="list_music_playlists",
            display_name="查看共享歌单",
            description="列出当前 OOPZ area 的共享歌单、ID 和歌曲数量。",
            input_model=EmptyToolInput,
            output_model=ListMusicPlaylistsOutput,
            effect=ToolEffect.READ,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=True,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        del arguments
        try:
            playlists = await self._playlists.list(context.identity)
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return ListMusicPlaylistsOutput(
            playlists=tuple(PlaylistSummaryOutput.from_summary(item) for item in playlists)
        )


class PlaylistIdInput(BaseModel):
    """Stable area-scoped playlist identity."""

    model_config = ConfigDict(extra="forbid")

    playlist_id: UUID = Field(description="歌单 ID；可先调用 list_music_playlists 获取")


class GetMusicPlaylistTool(_PlaylistTool):
    """Read one playlist and its ordered entry IDs."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="get_music_playlist",
            display_name="查看歌单内容",
            description="读取当前 OOPZ area 内一个共享歌单的有序歌曲和条目 ID。",
            input_model=PlaylistIdInput,
            output_model=MusicPlaylistOutput,
            effect=ToolEffect.READ,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=True,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        values = PlaylistIdInput.model_validate(arguments)
        try:
            playlist = await self._playlists.get(context.identity, values.playlist_id)
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return MusicPlaylistOutput.from_playlist(playlist)


class AddMusicPlaylistTrackInput(PlaylistIdInput):
    """Playlist target and catalog query for the appended song."""

    query: str = Field(min_length=1, max_length=200, description="要加入歌单的歌曲名和歌手")


class AddMusicPlaylistTrackOutput(BaseModel):
    """Appended entry metadata."""

    playlist_id: UUID
    entry: PlaylistEntryOutput


class AddMusicPlaylistTrackTool(_PlaylistTool):
    """Resolve the top catalog match and append it to a playlist."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="add_music_playlist_track",
            display_name="添加歌曲到歌单",
            description="搜索歌曲并把最佳匹配追加到当前 area 的指定共享歌单。",
            input_model=AddMusicPlaylistTrackInput,
            output_model=AddMusicPlaylistTrackOutput,
            effect=ToolEffect.WRITE,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=False,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        values = AddMusicPlaylistTrackInput.model_validate(arguments)
        try:
            entry = await self._playlists.add(
                context.identity,
                values.playlist_id,
                values.query,
            )
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return AddMusicPlaylistTrackOutput(
            playlist_id=values.playlist_id,
            entry=PlaylistEntryOutput.from_entry(entry),
        )


class RemoveMusicPlaylistTrackInput(PlaylistIdInput):
    """Playlist and entry identities for one deletion."""

    entry_id: UUID = Field(description="歌单条目 ID；可先调用 get_music_playlist 获取")


class RemoveMusicPlaylistTrackOutput(BaseModel):
    """Idempotent removal result."""

    playlist_id: UUID
    entry_id: UUID
    removed: bool


class RemoveMusicPlaylistTrackTool(_PlaylistTool):
    """Remove one entry and compact the persisted order."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="remove_music_playlist_track",
            display_name="从歌单移除歌曲",
            description="按条目 ID 从当前 area 的共享歌单移除一首歌。",
            input_model=RemoveMusicPlaylistTrackInput,
            output_model=RemoveMusicPlaylistTrackOutput,
            effect=ToolEffect.WRITE,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=False,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        values = RemoveMusicPlaylistTrackInput.model_validate(arguments)
        try:
            result = await self._playlists.remove(
                context.identity,
                values.playlist_id,
                values.entry_id,
            )
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return RemoveMusicPlaylistTrackOutput(
            playlist_id=result.playlist_id,
            entry_id=result.entry_id,
            removed=result.removed,
        )


class LoadMusicPlaylistOutput(BaseModel):
    """Playback queue replacement created from a shared playlist."""

    playlist_id: UUID
    playlist_name: str
    voice_channel_id: str
    loaded_count: int
    replaced_current: bool
    started_worker: bool


class LoadMusicPlaylistTool(_PlaylistTool):
    """Replace current playback and upcoming queue from a playlist."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="load_music_playlist",
            display_name="从歌单重建队列",
            description=("停止用户当前语音频道的现有播放，并按共享歌单顺序重建播放队列。"),
            input_model=PlaylistIdInput,
            output_model=LoadMusicPlaylistOutput,
            effect=ToolEffect.WRITE,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=False,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        values = PlaylistIdInput.model_validate(arguments)
        try:
            result = await self._playlists.load(context.identity, values.playlist_id)
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return LoadMusicPlaylistOutput(
            playlist_id=result.playlist_id,
            playlist_name=result.playlist_name,
            voice_channel_id=result.queue.voice_channel.channel_id,
            loaded_count=result.queue.loaded_count,
            replaced_current=result.queue.replaced_current,
            started_worker=result.queue.started_worker,
        )
