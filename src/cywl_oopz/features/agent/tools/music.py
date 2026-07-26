"""Agent tool adapters for the project-owned music request service."""

from __future__ import annotations

from typing import Never

from pydantic import BaseModel, ConfigDict, Field

from cywl_oopz.features.music.errors import (
    MusicCatalogError,
    MusicError,
    MusicNotFoundError,
    MusicPlaybackError,
    MusicQueryError,
    MusicQueueFullError,
    MusicVoiceChannelRequiredError,
)
from cywl_oopz.features.music.models import MusicQueueSnapshot, MusicTrack, QueuedTrack
from cywl_oopz.features.music.service import MusicRequestService

from .builtin import EmptyToolInput
from .models import (
    ToolDescriptor,
    ToolEffect,
    ToolExecutionContext,
    ToolExecutionError,
)


class _MusicTool:
    """Map expected music failures to stable, model-readable tool errors."""

    _ERROR_CODES = {
        MusicVoiceChannelRequiredError: "music_voice_channel_required",
        MusicQueryError: "invalid_music_query",
        MusicQueueFullError: "music_queue_full",
        MusicNotFoundError: "music_not_found",
        MusicCatalogError: "music_catalog_unavailable",
        MusicPlaybackError: "music_playback_failed",
    }

    @classmethod
    def _raise_tool_error(cls, error: MusicError) -> Never:
        for error_type, code in cls._ERROR_CODES.items():
            if isinstance(error, error_type):
                raise ToolExecutionError(code) from error
        raise ToolExecutionError("music_failed") from error


class MusicSearchInput(BaseModel):
    """A bounded natural-language catalog query."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200, description="歌曲名、歌手或两者组合")
    limit: int = Field(default=5, ge=1, le=10, description="返回候选数量")


class MusicTrackOutput(BaseModel):
    """Safe catalog metadata without an expiring stream URL."""

    source: str
    source_id: str
    title: str
    artists: tuple[str, ...]
    duration_ms: int | None

    @classmethod
    def from_track(cls, track: MusicTrack) -> MusicTrackOutput:
        return cls(
            source=track.source,
            source_id=track.source_id,
            title=track.title,
            artists=track.artists,
            duration_ms=track.duration_ms,
        )


class MusicSearchOutput(BaseModel):
    """Ordered music search candidates."""

    tracks: tuple[MusicTrackOutput, ...]


class SearchMusicCatalogTool(_MusicTool):
    """Search without mutating a voice-channel queue."""

    def __init__(
        self,
        music: MusicRequestService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._music = music
        self._descriptor = ToolDescriptor(
            name="search_music_catalog",
            description="按歌曲名或歌手搜索音乐目录，返回候选但不点歌。",
            input_model=MusicSearchInput,
            output_model=MusicSearchOutput,
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
        values = MusicSearchInput.model_validate(arguments)
        try:
            tracks = await self._music.search(values.query, limit=values.limit)
        except MusicError as exc:
            self._raise_tool_error(exc)
        return MusicSearchOutput(
            tracks=tuple(MusicTrackOutput.from_track(track) for track in tracks)
        )


class EnqueueMusicInput(BaseModel):
    """A query whose top match will be queued."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200, description="要点播的歌曲名和可选歌手")


class EnqueueMusicOutput(BaseModel):
    """Committed queue mutation result."""

    voice_channel_id: str
    position: int
    track: MusicTrackOutput


class EnqueueMusicTool(_MusicTool):
    """Queue the top result for the caller's real OOPZ voice channel."""

    def __init__(
        self,
        music: MusicRequestService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._music = music
        self._descriptor = ToolDescriptor(
            name="enqueue_music",
            description=("为用户当前所在的 OOPZ 语音频道点歌；用户不在语音频道时会失败。"),
            input_model=EnqueueMusicInput,
            output_model=EnqueueMusicOutput,
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
        values = EnqueueMusicInput.model_validate(arguments)
        try:
            result = await self._music.enqueue(
                context.identity,
                values.query,
                idempotency_key=(f"{context.run_id}:{values.query.strip().casefold()}"),
            )
        except MusicError as exc:
            self._raise_tool_error(exc)
        return EnqueueMusicOutput(
            voice_channel_id=result.voice_channel.channel_id,
            position=result.position,
            track=MusicTrackOutput.from_track(result.item.track),
        )


class QueuedMusicOutput(BaseModel):
    """Queue item metadata and requester."""

    id: str
    requested_by: str
    track: MusicTrackOutput

    @classmethod
    def from_item(cls, item: QueuedTrack) -> QueuedMusicOutput:
        return cls(
            id=str(item.id),
            requested_by=item.requested_by,
            track=MusicTrackOutput.from_track(item.track),
        )


class MusicQueueOutput(BaseModel):
    """Bounded queue state visible to the model."""

    voice_channel_id: str
    state: str
    current: QueuedMusicOutput | None
    upcoming: tuple[QueuedMusicOutput, ...]
    revision: int

    @classmethod
    def from_snapshot(cls, snapshot: MusicQueueSnapshot) -> MusicQueueOutput:
        return cls(
            voice_channel_id=snapshot.voice_channel.channel_id,
            state=snapshot.state.value,
            current=(
                QueuedMusicOutput.from_item(snapshot.current)
                if snapshot.current is not None
                else None
            ),
            upcoming=tuple(QueuedMusicOutput.from_item(item) for item in snapshot.upcoming),
            revision=snapshot.revision,
        )


class GetMusicQueueTool(_MusicTool):
    """Inspect the caller's current voice-channel queue."""

    def __init__(
        self,
        music: MusicRequestService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._music = music
        self._descriptor = ToolDescriptor(
            name="get_music_queue",
            description="查看用户当前语音频道的正在播放歌曲和后续队列。",
            input_model=EmptyToolInput,
            output_model=MusicQueueOutput,
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
            snapshot = await self._music.queue(context.identity)
        except MusicError as exc:
            self._raise_tool_error(exc)
        return MusicQueueOutput.from_snapshot(snapshot)


class MusicControlOutput(BaseModel):
    """Result of one idempotent playback-control request."""

    applied: bool


class _MusicControlTool(_MusicTool):
    """Shared descriptor plumbing for simple identity-scoped controls."""

    name: str
    description: str

    def __init__(
        self,
        music: MusicRequestService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._music = music
        self._descriptor = ToolDescriptor(
            name=self.name,
            description=self.description,
            input_model=EmptyToolInput,
            output_model=MusicControlOutput,
            effect=ToolEffect.WRITE,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=False,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor


class SkipMusicTool(_MusicControlTool):
    """Skip the current track."""

    name = "skip_music"
    description = "跳过用户当前语音频道正在播放的歌曲；没有歌曲时返回未执行。"

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        del arguments
        try:
            applied = await self._music.skip(context.identity)
        except MusicError as exc:
            self._raise_tool_error(exc)
        return MusicControlOutput(applied=applied)


class PauseMusicTool(_MusicControlTool):
    """Pause the current track."""

    name = "pause_music"
    description = "暂停用户当前语音频道正在播放的歌曲。"

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        del arguments
        try:
            applied = await self._music.pause(context.identity)
        except MusicError as exc:
            self._raise_tool_error(exc)
        return MusicControlOutput(applied=applied)


class ResumeMusicTool(_MusicControlTool):
    """Resume the current track."""

    name = "resume_music"
    description = "恢复用户当前语音频道已暂停的歌曲。"

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        del arguments
        try:
            applied = await self._music.resume(context.identity)
        except MusicError as exc:
            self._raise_tool_error(exc)
        return MusicControlOutput(applied=applied)
