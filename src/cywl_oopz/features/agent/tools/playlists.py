"""Agent tool adapters for area-shared music playlists."""

from __future__ import annotations

from typing import Never
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.features.music.errors import (
    MusicAreaRequiredError,
    MusicAuthenticationRequiredError,
    MusicCatalogError,
    MusicError,
    MusicExtractionTimeoutError,
    MusicGeoRestrictedError,
    MusicLiveUnsupportedError,
    MusicNoAudioFormatError,
    MusicNotFoundError,
    MusicPlaybackError,
    MusicPlaylistConflictError,
    MusicPlaylistEmptyError,
    MusicPlaylistFullError,
    MusicPlaylistNameError,
    MusicPlaylistNotFoundError,
    MusicQueryError,
    MusicQueueFullError,
    MusicReferenceError,
    MusicSourceDisabledError,
    MusicSourceRateLimitedError,
    MusicSourceUnavailableError,
    MusicTrackTooLongError,
    MusicUnsupportedContentError,
    MusicVoiceBusyError,
    MusicVoiceChannelRequiredError,
    NeteasePlaylistIncompleteError,
    NeteasePlaylistNotFoundError,
    NeteasePlaylistReferenceError,
    NeteasePlaylistTooLargeError,
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
from .music import (
    MusicSourceSelection,
    MusicTrackOutput,
    MusicTrackReferenceInput,
)


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
        MusicReferenceError: "invalid_music_reference",
        MusicSourceDisabledError: "music_source_disabled",
        MusicExtractionTimeoutError: "music_extraction_timeout",
        MusicSourceRateLimitedError: "music_rate_limited",
        MusicSourceUnavailableError: "music_source_unavailable",
        MusicAuthenticationRequiredError: "music_authentication_required",
        MusicGeoRestrictedError: "music_geo_restricted",
        MusicLiveUnsupportedError: "music_live_unsupported",
        MusicTrackTooLongError: "music_track_too_long",
        MusicNoAudioFormatError: "music_no_audio_format",
        MusicUnsupportedContentError: "music_content_unsupported",
        MusicVoiceBusyError: "music_voice_busy",
        MusicPlaybackError: "music_playback_failed",
        MusicCatalogError: "music_catalog_unavailable",
        NeteasePlaylistReferenceError: "invalid_netease_playlist_reference",
        NeteasePlaylistNotFoundError: "netease_playlist_not_found",
        NeteasePlaylistIncompleteError: "netease_playlist_incomplete",
        NeteasePlaylistTooLargeError: "netease_playlist_too_large",
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
    """Playlist target and exactly one query/URL or exact reference."""

    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_048,
        description="要加入歌单的关键词或受支持单曲 URL",
    )
    source: MusicSourceSelection = Field(
        default=MusicSourceSelection.AUTO,
        description="query 为普通文字时使用的来源",
    )
    track: MusicTrackReferenceInput | None = Field(
        default=None,
        description="先前搜索返回的精确 source/source_id",
    )

    @model_validator(mode="after")
    def require_one_target(self) -> AddMusicPlaylistTrackInput:
        if (self.query is None) == (self.track is None):
            raise ValueError("query 和 track 必须且只能提供一个")
        if self.track is not None and self.source is not MusicSourceSelection.AUTO:
            raise ValueError("精确 track 不再接受 source 选择")
        return self


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
            description=(
                "把一个关键词/单曲 URL 或先前搜索得到的精确歌曲追加到当前 area 共享歌单。"
            ),
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
            if values.track is not None:
                entry = await self._playlists.add_reference(
                    context.identity,
                    values.playlist_id,
                    values.track.reference,
                )
            else:
                assert values.query is not None
                entry = await self._playlists.add_input(
                    context.identity,
                    values.playlist_id,
                    values.query,
                    source=values.source.source_kind,
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


class RenameMusicPlaylistInput(PlaylistIdInput):
    """Area playlist target and normalized display name."""

    name: str = Field(min_length=1, max_length=80, description="新的共享歌单名称")


class RenameMusicPlaylistOutput(BaseModel):
    """Idempotent shared playlist rename result."""

    playlist_id: UUID
    old_name: str
    new_name: str
    changed: bool


class RenameMusicPlaylistTool(_PlaylistTool):
    """Rename one shared playlist without changing its entries."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="rename_music_playlist",
            display_name="重命名共享歌单",
            description="重命名当前 OOPZ area 的共享歌单，不改变其中歌曲。",
            input_model=RenameMusicPlaylistInput,
            output_model=RenameMusicPlaylistOutput,
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
        values = RenameMusicPlaylistInput.model_validate(arguments)
        try:
            result = await self._playlists.rename(
                context.identity,
                values.playlist_id,
                values.name,
            )
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return RenameMusicPlaylistOutput(
            playlist_id=result.playlist_id,
            old_name=result.old_name,
            new_name=result.new_name,
            changed=result.changed,
        )


class DeleteMusicPlaylistOutput(BaseModel):
    """Idempotent shared playlist deletion result."""

    playlist_id: UUID
    name: str | None
    deleted: bool
    removed_track_count: int


class DeleteMusicPlaylistTool(_PlaylistTool):
    """Delete a shared playlist and all of its persisted entries."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="delete_music_playlist",
            display_name="删除共享歌单",
            description=(
                "删除当前 OOPZ area 的共享歌单及其数据库条目；"
                "不会改变已经从该歌单载入的临时播放队列。"
            ),
            input_model=PlaylistIdInput,
            output_model=DeleteMusicPlaylistOutput,
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
            result = await self._playlists.delete(context.identity, values.playlist_id)
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return DeleteMusicPlaylistOutput(
            playlist_id=result.playlist_id,
            name=result.name,
            deleted=result.deleted,
            removed_track_count=result.removed_track_count,
        )


class ClearMusicPlaylistOutput(BaseModel):
    """Shared playlist entry clear result that preserves the playlist."""

    playlist_id: UUID
    name: str
    removed_track_count: int


class ClearMusicPlaylistTool(_PlaylistTool):
    """Clear persisted entries while preserving the area playlist."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="clear_music_playlist",
            display_name="清空共享歌单",
            description=(
                "清空当前 OOPZ area 共享歌单的 PostgreSQL 歌曲条目，但保留歌单本身；"
                "不会清空临时播放队列。"
            ),
            input_model=PlaylistIdInput,
            output_model=ClearMusicPlaylistOutput,
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
            result = await self._playlists.clear(context.identity, values.playlist_id)
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return ClearMusicPlaylistOutput(
            playlist_id=result.playlist_id,
            name=result.name,
            removed_track_count=result.removed_track_count,
        )


class NeteasePlaylistReferenceInput(BaseModel):
    """A numeric Netease playlist ID or canonical music.163.com URL."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(
        min_length=1,
        max_length=500,
        description="网易云歌单数字 ID 或包含 id 参数的 music.163.com 歌单链接",
    )


class PreviewNeteasePlaylistOutput(BaseModel):
    """Bounded source metadata shown before an area playlist mutation."""

    source_id: str
    name: str
    declared_track_count: int
    visible_track_count: int
    complete: bool
    requires_partial_confirmation: bool
    preview_tracks: tuple[MusicTrackOutput, ...]
    preview_truncated: bool


class PreviewNeteasePlaylistTool(_PlaylistTool):
    """Inspect a Netease playlist and expose incompleteness before import."""

    preview_limit = 10

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="preview_netease_playlist",
            display_name="预览网易云歌单",
            description=(
                "读取网易云歌单名称、歌曲总数和可导入数量。"
                "导入前必须先调用，以识别私有、缺歌或超过容量的歌单。"
            ),
            input_model=NeteasePlaylistReferenceInput,
            output_model=PreviewNeteasePlaylistOutput,
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
        del context
        values = NeteasePlaylistReferenceInput.model_validate(arguments)
        try:
            playlist = await self._playlists.preview_netease(values.reference)
        except MusicError as exc:
            self._raise_tool_error(exc)
        preview = playlist.tracks[: self.preview_limit]
        return PreviewNeteasePlaylistOutput(
            source_id=playlist.source_id,
            name=playlist.name,
            declared_track_count=playlist.declared_track_count,
            visible_track_count=playlist.loaded_track_count,
            complete=playlist.complete,
            requires_partial_confirmation=not playlist.complete,
            preview_tracks=tuple(MusicTrackOutput.from_track(track) for track in preview),
            preview_truncated=playlist.loaded_track_count > len(preview),
        )


class ImportNeteasePlaylistInput(NeteasePlaylistReferenceInput):
    """Source, optional area name, and explicit partial-import consent."""

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description="可选的 area 歌单名称；省略时沿用网易云歌单名称",
    )
    allow_partial: bool = Field(
        default=False,
        description="仅当用户明确同意缺歌或容量截断后才能设为 true",
    )


class ImportNeteasePlaylistOutput(BaseModel):
    """Atomic area playlist import result."""

    source_id: str
    source_name: str
    declared_track_count: int
    imported_track_count: int
    skipped_track_count: int
    partial: bool
    playlist: PlaylistSummaryOutput


class ImportNeteasePlaylistTool(_PlaylistTool):
    """Atomically create an area playlist from the visible Netease tracks."""

    def __init__(
        self,
        playlists: MusicPlaylistService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._playlists = playlists
        self._descriptor = ToolDescriptor(
            name="import_netease_playlist",
            display_name="导入网易云歌单",
            description=(
                "把网易云歌单按原顺序原子导入为当前 area 的新共享歌单。"
                "默认拒绝缺歌或超过容量的部分导入。"
            ),
            input_model=ImportNeteasePlaylistInput,
            output_model=ImportNeteasePlaylistOutput,
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
        values = ImportNeteasePlaylistInput.model_validate(arguments)
        try:
            result = await self._playlists.import_netease(
                context.identity,
                values.reference,
                name=values.name,
                allow_partial=values.allow_partial,
            )
        except (MusicError, DatabaseError) as exc:
            self._raise_tool_error(exc)
        return ImportNeteasePlaylistOutput(
            source_id=result.source_id,
            source_name=result.source_name,
            declared_track_count=result.declared_track_count,
            imported_track_count=result.imported_track_count,
            skipped_track_count=result.skipped_track_count,
            partial=result.partial,
            playlist=PlaylistSummaryOutput.from_playlist(result.playlist),
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
