"""Direct OOPZ command facade for music playback and shared playlists."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from oopz_sdk.events.context import EventContext

from cywl_oopz.commands.router import ParsedCommand
from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.agent.models import AgentIdentity
from cywl_oopz.features.chat.models import ConversationKey

from .errors import (
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
    MusicVoiceBusyError,
    MusicVoiceChannelRequiredError,
    NeteasePlaylistIncompleteError,
    NeteasePlaylistNotFoundError,
    NeteasePlaylistReferenceError,
    NeteasePlaylistTooLargeError,
)
from .models import (
    MusicPlaylist,
    MusicPlaylistSummary,
    MusicQueueSnapshot,
    MusicTrack,
    NeteasePlaylistSnapshot,
    PlaybackOrder,
    RepeatPolicy,
)
from .playlists import MusicPlaylistService
from .service import MusicRequestService

logger = logging.getLogger(__name__)


class MusicCommandUsageError(ValueError):
    """Raised when a direct music command does not match its documented grammar."""


@dataclass(frozen=True, slots=True)
class MusicCommandRenderer:
    """Render bounded OOPZ-compatible command responses."""

    prefix: str
    queue_limit: int = 8
    playlist_limit: int = 15
    playlist_track_limit: int = 12
    track_title_limit: int = 72
    track_artists_limit: int = 48

    def usage(self) -> str:
        root = f"{self.prefix}music"
        return "\n".join(
            (
                "🎵 **Music 命令**",
                f"{root} play <歌曲/歌手> · 点歌",
                f"{root} search <关键词> · 搜索歌曲",
                f"{root} queue · 查看队列",
                f"{root} skip | pause | resume | clear",
                f"{root} mode [sequential|shuffle] [off|one|all]",
                f"{root} playlist list",
                f"{root} playlist show|load|delete|clear <#编号或UUID>",
                f"{root} playlist create <名称>",
                f"{root} playlist rename <歌单> <新名称>",
                f"{root} playlist add <歌单> <歌曲/歌手>",
                f"{root} playlist remove <歌单> <曲目序号或条目UUID>",
                f"{root} playlist preview <网易云歌单ID或链接>",
                f"{root} playlist import <ID或链接> [--partial] [新名称]",
                "歌单列表中的 #编号会在每次操作时重新解析。",
            )
        )

    def tracks(self, tracks: tuple[MusicTrack, ...]) -> str:
        if not tracks:
            return "没有找到匹配的歌曲。"
        lines = [f"🔎 **找到 {len(tracks)} 首歌曲**"]
        lines.extend(
            f"{index}. {self._track(track)}" for index, track in enumerate(tracks, start=1)
        )
        return "\n".join(lines)

    def queue(self, snapshot: MusicQueueSnapshot) -> str:
        state_names = {
            "idle": "空闲",
            "waiting": "等待播放",
            "loading": "正在加载",
            "playing": "正在播放",
            "paused": "已暂停",
            "recovering": "正在重连",
            "releasing": "正在退出语音",
            "failed": "播放中断",
        }
        order = "随机" if snapshot.policy.order is PlaybackOrder.SHUFFLE else "顺序"
        repeat = {
            RepeatPolicy.OFF: "不循环",
            RepeatPolicy.ONE: "单曲循环",
            RepeatPolicy.ALL: "列表循环",
        }[snapshot.policy.repeat]
        lines = [
            "🎵 **播放队列**",
            f"状态：{state_names[snapshot.state.value]} · {order} · {repeat}",
        ]
        if snapshot.current is not None:
            lines.append(f"▶ {self._track(snapshot.current.track)}")
        if snapshot.upcoming:
            lines.append("**接下来**")
            lines.extend(
                f"{index}. {self._track(item.track)}"
                for index, item in enumerate(
                    snapshot.upcoming[: self.queue_limit],
                    start=1,
                )
            )
            omitted = len(snapshot.upcoming) - self.queue_limit
            if omitted > 0:
                lines.append(f"… 还有 {omitted} 首")
        elif snapshot.current is None:
            lines.append("队列为空。")
        if snapshot.cycle_completed_count:
            lines.append(f"本轮已完成：{snapshot.cycle_completed_count} 首")
        if snapshot.last_failure is not None:
            failure = {
                "voice_left": "语音连接已断开",
                "backend_closed": "音频后端已关闭",
                "catalog_error": "歌曲地址解析失败",
                "track_error": "歌曲播放失败",
                "release_failed": "暂时无法退出语音频道",
            }.get(snapshot.last_failure.code.value, "播放发生错误")
            lines.append(f"⚠️ 最近错误：{failure}")
        return "\n".join(lines)

    def policy(self, order: PlaybackOrder, repeat: RepeatPolicy, *, changed: bool) -> str:
        order_name = "随机" if order is PlaybackOrder.SHUFFLE else "顺序"
        repeat_name = {
            RepeatPolicy.OFF: "不循环",
            RepeatPolicy.ONE: "单曲循环",
            RepeatPolicy.ALL: "列表循环",
        }[repeat]
        prefix = "播放策略已设置" if changed else "当前播放策略"
        return f"🎛️ **{prefix}** {order_name} · {repeat_name}"

    def playlists(self, playlists: tuple[MusicPlaylistSummary, ...]) -> str:
        if not playlists:
            return "当前 area 还没有共享歌单。"
        visible = playlists[: self.playlist_limit]
        lines = [f"📚 **共享歌单 · {len(playlists)} 个**"]
        lines.extend(
            f"#{index} **{playlist.name}** · {playlist.track_count} 首"
            for index, playlist in enumerate(visible, start=1)
        )
        if len(playlists) > len(visible):
            lines.append(f"… 还有 {len(playlists) - len(visible)} 个")
        return "\n".join(lines)

    def playlist(self, playlist: MusicPlaylist) -> str:
        lines = [f"📀 **{playlist.name}** · {playlist.track_count} 首"]
        visible = playlist.entries[: self.playlist_track_limit]
        lines.extend(f"{entry.position}. {self._track(entry.track)}" for entry in visible)
        if len(playlist.entries) > len(visible):
            lines.append(f"… 还有 {len(playlist.entries) - len(visible)} 首")
        if not playlist.entries:
            lines.append("歌单为空。")
        return "\n".join(lines)

    def preview(self, playlist: NeteasePlaylistSnapshot) -> str:
        lines = [
            f"☁️ **{playlist.name}**",
            f"可读取：{playlist.loaded_track_count}/{playlist.declared_track_count} 首",
            f"完整性：{'完整' if playlist.complete else '不完整，需要 --partial 确认'}",
        ]
        lines.extend(
            f"{index}. {self._track(track)}"
            for index, track in enumerate(
                playlist.tracks[: self.playlist_track_limit],
                start=1,
            )
        )
        if playlist.loaded_track_count > self.playlist_track_limit:
            lines.append(
                f"… 还有 {playlist.loaded_track_count - self.playlist_track_limit} 首可导入"
            )
        return "\n".join(lines)

    def _track(self, track: MusicTrack) -> str:
        title = self._bounded(track.title, self.track_title_limit)
        artists = self._bounded(" / ".join(track.artists), self.track_artists_limit)
        return f"**{title}** · {artists}" if artists else f"**{title}**"

    @staticmethod
    def _bounded(value: str, limit: int) -> str:
        normalized = " ".join(value.split())
        return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"


class MusicCommand:
    """Map one ``/music`` namespace onto existing playback and playlist services."""

    name = "music"
    description = "点歌并管理播放队列、播放模式和共享歌单。"

    _ORDER_ALIASES = {
        "sequential": PlaybackOrder.SEQUENTIAL,
        "顺序": PlaybackOrder.SEQUENTIAL,
        "shuffle": PlaybackOrder.SHUFFLE,
        "random": PlaybackOrder.SHUFFLE,
        "随机": PlaybackOrder.SHUFFLE,
    }
    _REPEAT_ALIASES = {
        "off": RepeatPolicy.OFF,
        "none": RepeatPolicy.OFF,
        "不循环": RepeatPolicy.OFF,
        "one": RepeatPolicy.ONE,
        "single": RepeatPolicy.ONE,
        "单曲": RepeatPolicy.ONE,
        "all": RepeatPolicy.ALL,
        "list": RepeatPolicy.ALL,
        "列表": RepeatPolicy.ALL,
    }

    def __init__(
        self,
        music: MusicRequestService,
        playlists: MusicPlaylistService,
        command_prefix: str,
        renderer: MusicCommandRenderer | None = None,
    ) -> None:
        self._music = music
        self._playlists = playlists
        self._renderer = renderer or MusicCommandRenderer(command_prefix)

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        try:
            identity = self._identity(context)
            if not command.arguments or command.arguments[0].casefold() in {"help", "帮助"}:
                await context.reply(self._renderer.usage())
                return
            action = command.arguments[0].casefold()
            arguments = command.arguments[1:]
            if action in {"play", "add", "点歌"}:
                await self._play(identity, arguments, context)
            elif action in {"search", "find", "搜索"}:
                await self._search(arguments, context)
            elif action in {"queue", "status", "队列"}:
                self._require_count(arguments, 0)
                await context.reply(self._renderer.queue(await self._music.queue(identity)))
            elif action in {"skip", "next", "下一首"}:
                self._require_count(arguments, 0)
                applied = await self._music.skip(identity)
                await context.reply("⏭️ 已切换下一首。" if applied else "当前没有正在播放的歌曲。")
            elif action in {"pause", "暂停"}:
                self._require_count(arguments, 0)
                applied = await self._music.pause(identity)
                await context.reply("⏸️ 已暂停播放。" if applied else "当前没有可暂停的歌曲。")
            elif action in {"resume", "继续"}:
                self._require_count(arguments, 0)
                applied = await self._music.resume(identity)
                await context.reply("▶ 已继续播放。" if applied else "当前没有已暂停的歌曲。")
            elif action in {"clear", "清空"}:
                self._require_count(arguments, 0)
                result = await self._music.clear(identity)
                stopped = "已停止当前歌曲，" if result.stopped_current else ""
                await context.reply(f"🧹 {stopped}已从播放队列移除 {result.removed_count} 首。")
            elif action in {"mode", "policy", "模式"}:
                await self._mode(identity, arguments, context)
            elif action in {"playlist", "playlists", "pl", "歌单"}:
                await self._playlist(identity, arguments, context)
            else:
                raise MusicCommandUsageError
        except MusicCommandUsageError:
            await context.reply(self._renderer.usage())
        except (MusicError, DatabaseError) as exc:
            logger.info(
                "Music command rejected: conversation=%s error=%s",
                self._conversation_ref(context),
                exception_kind(exc),
            )
            await context.reply(self._error_message(exc))
        except ValueError as exc:
            logger.info("Music command context invalid: error=%s", exception_kind(exc))
            await context.reply("无法识别发起人或当前频道。")
        except Exception as exc:
            logger.error(
                "Unexpected music command failure: conversation=%s error=%s",
                self._conversation_ref(context),
                exception_kind(exc),
                exc_info=True,
            )
            await context.reply("音乐操作失败，请稍后重试。")

    async def _play(
        self,
        identity: AgentIdentity,
        arguments: tuple[str, ...],
        context: EventContext,
    ) -> None:
        query = self._joined(arguments)
        result = await self._music.enqueue(
            identity,
            query,
            idempotency_key=(
                f"music-command:{identity.source_message_id}" if identity.source_message_id else ""
            ),
        )
        await context.reply(
            f"🎵 已加入 **{result.item.track.title}** · 队列第 {result.position} 位"
        )

    async def _search(self, arguments: tuple[str, ...], context: EventContext) -> None:
        tracks = await self._music.search(self._joined(arguments), limit=5)
        await context.reply(self._renderer.tracks(tracks))

    async def _mode(
        self,
        identity: AgentIdentity,
        arguments: tuple[str, ...],
        context: EventContext,
    ) -> None:
        if not arguments:
            snapshot = await self._music.queue(identity)
            await context.reply(
                self._renderer.policy(
                    snapshot.policy.order,
                    snapshot.policy.repeat,
                    changed=False,
                )
            )
            return
        order: PlaybackOrder | None = None
        repeat: RepeatPolicy | None = None
        for argument in arguments:
            token = argument.casefold()
            if token in {"order", "repeat", "顺序方式", "循环方式"}:
                continue
            if token in self._ORDER_ALIASES and order is None:
                order = self._ORDER_ALIASES[token]
            elif token in self._REPEAT_ALIASES and repeat is None:
                repeat = self._REPEAT_ALIASES[token]
            else:
                raise MusicCommandUsageError
        if order is None and repeat is None:
            raise MusicCommandUsageError
        result = await self._music.set_policy(identity, order=order, repeat=repeat)
        await context.reply(
            self._renderer.policy(
                result.policy.order,
                result.policy.repeat,
                changed=result.changed,
            )
        )

    async def _playlist(
        self,
        identity: AgentIdentity,
        arguments: tuple[str, ...],
        context: EventContext,
    ) -> None:
        if not arguments:
            await context.reply(self._renderer.playlists(await self._playlists.list(identity)))
            return
        action = arguments[0].casefold()
        values = arguments[1:]
        if action in {"list", "ls", "列表"}:
            self._require_count(values, 0)
            await context.reply(self._renderer.playlists(await self._playlists.list(identity)))
        elif action in {"show", "get", "查看"}:
            self._require_count(values, 1)
            playlist = await self._resolve_playlist(identity, values[0])
            await context.reply(self._renderer.playlist(playlist))
        elif action in {"create", "new", "新建"}:
            playlist = await self._playlists.create(identity, self._joined(values))
            await context.reply(f"✅ 已创建共享歌单 **{playlist.name}**。")
        elif action in {"rename", "重命名"}:
            self._require_minimum(values, 2)
            playlist = await self._resolve_playlist(identity, values[0])
            result = await self._playlists.rename(identity, playlist.id, self._joined(values[1:]))
            message = (
                f"✅ **{result.old_name}** → **{result.new_name}**"
                if result.changed
                else f"共享歌单已经叫 **{result.new_name}**。"
            )
            await context.reply(message)
        elif action in {"delete", "rm", "删除"}:
            self._require_count(values, 1)
            playlist_id = await self._resolve_playlist_id(identity, values[0])
            result = await self._playlists.delete(identity, playlist_id)
            if result.deleted:
                await context.reply(
                    f"🗑️ 已删除共享歌单 **{result.name}** · 移除 {result.removed_track_count} 首。"
                )
            else:
                await context.reply("这个共享歌单已经不存在。")
        elif action in {"clear", "清空"}:
            self._require_count(values, 1)
            playlist = await self._resolve_playlist(identity, values[0])
            result = await self._playlists.clear(identity, playlist.id)
            await context.reply(
                f"🧹 已清空共享歌单 **{result.name}** · 移除 {result.removed_track_count} 首。"
            )
        elif action in {"add", "添加"}:
            self._require_minimum(values, 2)
            playlist = await self._resolve_playlist(identity, values[0])
            entry = await self._playlists.add(identity, playlist.id, self._joined(values[1:]))
            await context.reply(
                f"✅ 已把 **{entry.track.title}** 加入 **{playlist.name}** · 第 {entry.position} 首"
            )
        elif action in {"remove", "移除"}:
            self._require_count(values, 2)
            playlist = await self._resolve_playlist(identity, values[0])
            entry_id = self._resolve_entry_id(playlist, values[1])
            result = await self._playlists.remove(identity, playlist.id, entry_id)
            await context.reply(
                "✅ 已从共享歌单移除歌曲。" if result.removed else "歌单中没有这个条目。"
            )
        elif action in {"load", "play", "播放"}:
            self._require_count(values, 1)
            playlist = await self._resolve_playlist(identity, values[0])
            result = await self._playlists.load(identity, playlist.id)
            await context.reply(
                f"▶ 已从 **{result.playlist_name}** 重建播放队列 · {result.queue.loaded_count} 首"
            )
        elif action in {"preview", "预览"}:
            self._require_count(values, 1)
            await context.reply(
                self._renderer.preview(await self._playlists.preview_netease(values[0]))
            )
        elif action in {"import", "导入"}:
            self._require_minimum(values, 1)
            reference = values[0]
            options = list(values[1:])
            allow_partial = "--partial" in options
            options = [value for value in options if value != "--partial"]
            if any(value.startswith("--") for value in options):
                raise MusicCommandUsageError
            result = await self._playlists.import_netease(
                identity,
                reference,
                name=" ".join(options).strip() or None,
                allow_partial=allow_partial,
            )
            suffix = (
                f" · 跳过 {result.skipped_track_count} 首" if result.skipped_track_count else ""
            )
            await context.reply(
                f"☁️ 已导入为 **{result.playlist.name}** · {result.imported_track_count} 首{suffix}"
            )
        else:
            raise MusicCommandUsageError

    async def _resolve_playlist(
        self,
        identity: AgentIdentity,
        reference: str,
    ) -> MusicPlaylist:
        return await self._playlists.get(
            identity,
            await self._resolve_playlist_id(identity, reference),
        )

    async def _resolve_playlist_id(self, identity: AgentIdentity, reference: str) -> UUID:
        try:
            return UUID(reference)
        except ValueError:
            pass
        normalized = reference.removeprefix("#")
        if not normalized.isdecimal() or int(normalized) < 1:
            raise MusicCommandUsageError
        playlists = await self._playlists.list(identity)
        index = int(normalized) - 1
        if index >= len(playlists):
            raise MusicPlaylistNotFoundError("Playlist index is outside the current area list")
        return playlists[index].id

    @staticmethod
    def _resolve_entry_id(playlist: MusicPlaylist, reference: str) -> UUID:
        try:
            return UUID(reference)
        except ValueError:
            pass
        normalized = reference.removeprefix("#")
        if not normalized.isdecimal() or int(normalized) < 1:
            raise MusicCommandUsageError
        position = int(normalized)
        entry = next((item for item in playlist.entries if item.position == position), None)
        if entry is None:
            raise MusicPlaylistNotFoundError("Playlist entry position does not exist")
        return entry.id

    @staticmethod
    def _identity(context: EventContext) -> AgentIdentity:
        key = ConversationKey.from_oopz_context(context)
        message = getattr(getattr(context, "event", None), "message", None)
        return AgentIdentity(
            key.person_id,
            key,
            source_message_id=str(getattr(message, "message_id", "")).strip(),
            transport_channel_id=str(getattr(message, "channel", "")).strip(),
        )

    @staticmethod
    def _joined(arguments: tuple[str, ...]) -> str:
        value = " ".join(arguments).strip()
        if not value:
            raise MusicCommandUsageError
        return value

    @staticmethod
    def _require_count(arguments: tuple[str, ...], count: int) -> None:
        if len(arguments) != count:
            raise MusicCommandUsageError

    @staticmethod
    def _require_minimum(arguments: tuple[str, ...], count: int) -> None:
        if len(arguments) < count:
            raise MusicCommandUsageError

    @staticmethod
    def _conversation_ref(context: EventContext) -> str:
        try:
            key = ConversationKey.from_oopz_context(context)
        except ValueError:
            return opaque_ref("unknown")
        return opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id)

    def _error_message(self, error: Exception) -> str:
        if isinstance(error, MusicVoiceChannelRequiredError):
            return "请先加入当前 area 的语音频道。"
        if isinstance(error, MusicVoiceBusyError):
            return "语音频道正被实时对话或其他功能使用，请稍后再试。"
        if isinstance(error, MusicQueueFullError):
            return "播放队列已满，请先播放、跳过或清空一些歌曲。"
        if isinstance(error, MusicNotFoundError):
            return "没有找到可用的歌曲。"
        if isinstance(error, MusicCatalogError):
            return "音乐目录暂时不可用，请稍后重试。"
        if isinstance(error, MusicPlaylistConflictError):
            return "当前 area 已有同名共享歌单。"
        if isinstance(error, MusicPlaylistNotFoundError):
            return "当前 area 没有这个共享歌单或条目，请重新查看歌单列表。"
        if isinstance(error, MusicPlaylistFullError):
            return "这个共享歌单已满。"
        if isinstance(error, MusicPlaylistEmptyError):
            return "这个共享歌单还是空的，暂时不能载入播放。"
        if isinstance(error, MusicPlaylistNameError):
            return "歌单名称不能为空，且最多 80 个字符。"
        if isinstance(error, MusicAreaRequiredError):
            return "共享歌单命令只能在 OOPZ area 频道中使用。"
        if isinstance(error, NeteasePlaylistReferenceError):
            return "无法识别网易云歌单 ID 或链接。"
        if isinstance(error, NeteasePlaylistNotFoundError):
            return "没有找到这个网易云歌单，或它当前不可见。"
        if isinstance(error, (NeteasePlaylistIncompleteError, NeteasePlaylistTooLargeError)):
            return (
                "这个网易云歌单只能部分导入。请先 preview，确认后在 import 命令中加入 --partial。"
            )
        if isinstance(error, MusicQueryError):
            return "音乐命令参数不正确，请缩短关键词或检查播放模式。"
        if isinstance(error, MusicPlaybackError):
            return "音乐播放操作失败，请稍后重试。"
        if isinstance(error, DatabaseError):
            return "共享歌单服务暂时不可用，请稍后重试。"
        return "音乐操作失败，请稍后重试。"
