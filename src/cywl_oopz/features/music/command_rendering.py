"""Bounded OOPZ rendering for direct music commands."""

from __future__ import annotations

from dataclasses import dataclass

from .models import (
    MusicPlaylist,
    MusicPlaylistSummary,
    MusicProviderHealth,
    MusicProviderHealthState,
    MusicQueueSnapshot,
    MusicSourceKind,
    MusicTrack,
    NeteasePlaylistSnapshot,
    PlaybackOrder,
    RepeatPolicy,
)


@dataclass(frozen=True, slots=True)
class MusicCommandRenderer:
    """Render bounded OOPZ-compatible command responses."""

    prefix: str
    queue_limit: int = 8
    playlist_limit: int = 15
    playlist_track_limit: int = 12
    track_title_limit: int = 72
    track_artists_limit: int = 48

    _SOURCE_NAMES = {
        MusicSourceKind.NETEASE: ("☁️", "网易云"),
        MusicSourceKind.YOUTUBE: ("▶️", "YouTube"),
        MusicSourceKind.BILIBILI: ("📺", "Bilibili"),
    }

    def usage(self) -> str:
        root = f"{self.prefix}music"
        return "\n".join(
            (
                "🎵 **Music 命令**",
                f"{root} play [--source 来源] <关键词或单曲URL>",
                f"{root} search [--source 来源] <关键词>",
                f"{root} sources · 查看来源状态",
                f"{root} queue · 查看队列",
                f"{root} skip | pause | resume | clear",
                f"{root} mode [sequential|shuffle] [off|one|all]",
                f"{root} playlist list",
                f"{root} playlist show|load|delete|clear <#编号或UUID>",
                f"{root} playlist create <名称>",
                f"{root} playlist rename <歌单> <新名称>",
                f"{root} playlist add <歌单> [--source 来源] <关键词或URL>",
                f"{root} playlist remove <歌单> <曲目序号或条目UUID>",
                f"{root} playlist preview <网易云歌单ID或链接>",
                f"{root} playlist import <ID或链接> [--partial] [新名称]",
                "来源：auto、netease、youtube、bilibili；--source 必须放在内容前。",
                "歌单列表中的 #编号会在每次操作时重新解析。",
            )
        )

    def tracks(self, tracks: tuple[MusicTrack, ...]) -> str:
        if not tracks:
            return "没有找到匹配的歌曲。"
        sources = {track.source for track in tracks}
        scope = self.source_name(next(iter(sources))) if len(sources) == 1 else "多来源"
        lines = [f"🔎 **{scope} · 找到 {len(tracks)} 首**"]
        lines.extend(f"{index}. {self.track(track)}" for index, track in enumerate(tracks, start=1))
        return "\n".join(lines)

    def sources(
        self,
        health: tuple[MusicProviderHealth, ...],
        *,
        default_source: MusicSourceKind,
    ) -> str:
        lines = ["🎵 **音乐来源**"]
        for item in health:
            status = {
                MusicProviderHealthState.READY: "✅",
                MusicProviderHealthState.DEGRADED: "⚠️",
                MusicProviderHealthState.UNAVAILABLE: "❌",
                MusicProviderHealthState.AUTHENTICATION_REQUIRED: "🔐",
            }[item.state]
            default = " · 默认" if item.source is default_source else ""
            detail = f" · {self._bounded(item.detail, 80)}" if item.detail else ""
            lines.append(f"{status} {self.source_name(item.source)}{default}{detail}")
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
            lines.append(f"▶ {self.track(snapshot.current.track)}")
        if snapshot.upcoming:
            lines.append("**接下来**")
            lines.extend(
                f"{index}. {self.track(item.track)}"
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
        lines.extend(f"{entry.position}. {self.track(entry.track)}" for entry in visible)
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
            f"{index}. {self.track(track)}"
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

    def track(self, track: MusicTrack) -> str:
        title = self._bounded(track.title, self.track_title_limit)
        artists = self._bounded(" / ".join(track.artists), self.track_artists_limit)
        marker = self.source_marker(track.source)
        return f"{marker} **{title}** · {artists}" if artists else f"{marker} **{title}**"

    @classmethod
    def source_marker(cls, source: MusicSourceKind) -> str:
        return cls._SOURCE_NAMES[MusicSourceKind(source)][0]

    @classmethod
    def source_name(cls, source: MusicSourceKind) -> str:
        return cls._SOURCE_NAMES[MusicSourceKind(source)][1]

    @staticmethod
    def _bounded(value: str, limit: int) -> str:
        normalized = " ".join(value.split())
        return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"
