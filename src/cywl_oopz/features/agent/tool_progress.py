"""Project-owned, display-safe projections for Agent tool progress."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

_WHITESPACE = re.compile(r"\s+")
_LINE_BREAKS = re.compile(r"[\r\n]+")


@dataclass(frozen=True, slots=True)
class ToolProgressPresentation:
    """Structured display data independent from OOPZ message formatting."""

    subject: str = ""
    summary: str = ""
    items: tuple[str, ...] = ()
    preview_lines: tuple[str, ...] = ()


class ToolProgressProjector(Protocol):
    """Select useful, safe presentation data for one tool family."""

    def request(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        """Project validated model arguments."""

    def result(
        self,
        tool_name: str,
        values: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        """Project a successful tool result."""


class _ProjectionSupport:
    """Shared bounded text helpers for trusted project projectors."""

    _sensitive_name_parts = frozenset(
        {
            "api_key",
            "authorization",
            "cookie",
            "credential",
            "password",
            "secret",
            "token",
        }
    )

    @classmethod
    def scalar(cls, value: object, *, limit: int = 80) -> str:
        if value is None or (isinstance(value, Mapping | Sequence) and not isinstance(value, str)):
            return ""
        rendered = _WHITESPACE.sub(" ", str(value)).strip()
        if not rendered:
            return ""
        return rendered[: limit - 1] + "…" if len(rendered) > limit else rendered

    @classmethod
    def line(cls, value: object, *, limit: int) -> str:
        rendered = _LINE_BREAKS.sub(" ", str(value))
        return cls.scalar(rendered, limit=limit)

    @classmethod
    def host(cls, value: object) -> str:
        url = cls.scalar(value, limit=180)
        if not url:
            return ""
        try:
            parsed = urlsplit(url)
        except ValueError:
            return url
        return parsed.netloc or parsed.path or url

    @classmethod
    def sensitive(cls, name: str) -> bool:
        normalized = name.casefold()
        return any(part in normalized for part in cls._sensitive_name_parts)

    @classmethod
    def preview(cls, value: object, *, limit: int = 3) -> tuple[str, ...]:
        if not isinstance(value, str):
            return ()
        lines: list[str] = []
        seen: set[str] = set()
        for raw_line in value.splitlines():
            line = cls.line(raw_line, limit=120)
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)
            if len(lines) == limit:
                break
        return tuple(lines)


class WebSearchProgressProjector(_ProjectionSupport):
    """Present web queries and a small set of usable result links."""

    def request(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        del tool_name
        query = self.scalar(arguments.get("query"))
        return ToolProgressPresentation(subject=f"「{query}」" if query else "")

    def result(
        self,
        tool_name: str,
        values: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        del tool_name
        raw_results = values.get("results")
        results = (
            raw_results
            if isinstance(raw_results, Sequence) and not isinstance(raw_results, str)
            else ()
        )
        urls: list[str] = []
        for result in results:
            if not isinstance(result, Mapping):
                continue
            url = self.line(result.get("url", ""), limit=180)
            if url and url not in urls:
                urls.append(url)
            if len(urls) == 3:
                break
        return ToolProgressPresentation(
            summary=f"找到 {len(results)} 条结果",
            items=tuple(urls),
        )


class BrowserProgressProjector(_ProjectionSupport):
    """Present page identity and bounded, meaningful page previews."""

    def request(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        if "url" in arguments:
            return ToolProgressPresentation(subject=self.host(arguments.get("url")))
        if tool_name == "browser_wait":
            condition = next(
                (
                    arguments.get(name)
                    for name in ("text", "selector", "load_state", "milliseconds")
                    if arguments.get(name) is not None
                ),
                None,
            )
            return ToolProgressPresentation(subject=self.scalar(condition))
        for name in ("ref", "key"):
            value = self.scalar(arguments.get(name))
            if value:
                return ToolProgressPresentation(subject=value)
        return ToolProgressPresentation()

    def result(
        self,
        tool_name: str,
        values: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        if tool_name == "browser_close":
            return ToolProgressPresentation(
                summary=("浏览器已关闭" if values.get("closed") else "没有活动的浏览器")
            )
        if tool_name == "browser_fill":
            return ToolProgressPresentation(
                summary=("输入框已填写" if values.get("applied") else "输入框未改变")
            )
        title = self.scalar(values.get("title"), limit=100)
        truncated = values.get("truncated") is True
        if tool_name == "read_web_page":
            content = values.get("content")
            summary = title or "网页读取完成"
            if truncated:
                summary = f"{summary} · 内容已截断"
            return ToolProgressPresentation(
                summary=summary,
                preview_lines=self.preview(content),
            )
        snapshot = values.get("snapshot")
        summary = title or "页面状态已更新"
        if truncated:
            summary = f"{summary} · 内容已截断"
        return ToolProgressPresentation(
            summary=summary,
            preview_lines=self.preview(snapshot, limit=2),
        )


class MusicProgressProjector(_ProjectionSupport):
    """Present music queries and compact queue outcomes."""

    _source_names = {
        "netease": "网易云",
        "youtube": "YouTube",
        "bilibili": "Bilibili",
    }

    def request(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        track = arguments.get("track")
        track_values = track if isinstance(track, Mapping) else {}
        target = (
            arguments.get("query")
            or arguments.get("name")
            or arguments.get("reference")
            or track_values.get("source_id")
        )
        source = self._source_name(
            track_values.get("source") or arguments.get("source"),
            target=target,
        )
        value = self.scalar(target)
        if tool_name == "search_music_catalog" and value:
            value = f"「{value}」"
        subject = self._join(source, value)
        return ToolProgressPresentation(subject=subject)

    def result(
        self,
        tool_name: str,
        values: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        if tool_name == "search_music_catalog":
            tracks = values.get("tracks")
            count = (
                len(tracks) if isinstance(tracks, Sequence) and not isinstance(tracks, str) else 0
            )
            source = self._tracks_source_name(tracks)
            return ToolProgressPresentation(summary=self._join(source, f"找到 {count} 首歌曲"))
        if tool_name == "get_music_queue":
            upcoming = values.get("upcoming")
            count = (
                len(upcoming)
                if isinstance(upcoming, Sequence) and not isinstance(upcoming, str)
                else 0
            )
            current = values.get("current")
            order = "随机" if values.get("order") == "shuffle" else "顺序"
            repeat_names = {"off": "不循环", "one": "单曲循环", "all": "列表循环"}
            repeat = repeat_names.get(str(values.get("repeat", "")), "不循环")
            policy = f"{order} · {repeat}"
            if values.get("state") == "failed":
                failure = values.get("last_failure")
                failure_code = failure.get("code") if isinstance(failure, Mapping) else ""
                failure_names = {
                    "voice_left": "语音连接已断开",
                    "backend_closed": "音频后端已关闭",
                    "catalog_error": "歌曲地址解析失败",
                    "track_error": "歌曲播放失败",
                    "release_failed": "暂时无法退出语音频道",
                }
                reason = failure_names.get(str(failure_code), "播放发生错误")
                return ToolProgressPresentation(summary=f"播放已中断 · 保留 {count} 首 · {reason}")
            if current:
                summary = f"正在播放 · 后续 {count} 首 · {policy}"
            elif count:
                summary = f"当前未播放 · 后续 {count} 首 · {policy}"
            else:
                summary = "当前未播放 · 队列为空"
            return ToolProgressPresentation(summary=summary)
        if tool_name == "set_music_playback_mode":
            order = "随机" if values.get("order") == "shuffle" else "顺序"
            repeat_names = {"off": "不循环", "one": "单曲循环", "all": "列表循环"}
            repeat = repeat_names.get(str(values.get("repeat", "")), "不循环")
            policy = f"{order} · {repeat}"
            return ToolProgressPresentation(
                summary=f"{policy}已设置" if values.get("changed") else f"已是{policy}"
            )
        if tool_name == "create_music_playlist":
            playlist = values.get("playlist")
            name = self.scalar(playlist.get("name")) if isinstance(playlist, Mapping) else ""
            return ToolProgressPresentation(summary=f"歌单「{name}」已创建")
        if tool_name == "list_music_playlists":
            playlists = values.get("playlists")
            count = (
                len(playlists)
                if isinstance(playlists, Sequence) and not isinstance(playlists, str)
                else 0
            )
            return ToolProgressPresentation(summary=f"找到 {count} 个共享歌单")
        if tool_name == "get_music_playlist":
            name = self.scalar(values.get("name"))
            entries = values.get("entries")
            count = (
                len(entries)
                if isinstance(entries, Sequence) and not isinstance(entries, str)
                else 0
            )
            return ToolProgressPresentation(summary=f"歌单「{name}」· {count} 首")
        if tool_name == "add_music_playlist_track":
            entry = values.get("entry")
            track = entry.get("track") if isinstance(entry, Mapping) else None
            title = self.scalar(track.get("title")) if isinstance(track, Mapping) else ""
            source = self._track_source_name(track)
            return ToolProgressPresentation(
                summary=self._join(source, f"歌曲「{title}」已加入歌单")
            )
        if tool_name == "remove_music_playlist_track":
            return ToolProgressPresentation(
                summary=("歌曲已移出歌单" if values.get("removed") else "歌单中没有该条目")
            )
        if tool_name == "rename_music_playlist":
            old_name = self.scalar(values.get("old_name"))
            new_name = self.scalar(values.get("new_name"))
            summary = (
                f"「{old_name}」→「{new_name}」"
                if values.get("changed")
                else f"已经叫「{new_name}」"
            )
            return ToolProgressPresentation(summary=summary)
        if tool_name == "delete_music_playlist":
            name = self.scalar(values.get("name"))
            if not values.get("deleted"):
                return ToolProgressPresentation(summary="共享歌单已不存在")
            count = values.get("removed_track_count")
            count = count if isinstance(count, int) else 0
            suffix = f" · 同时移除 {count} 首" if count else ""
            return ToolProgressPresentation(summary=f"歌单「{name}」已删除{suffix}")
        if tool_name == "clear_music_playlist":
            name = self.scalar(values.get("name"))
            count = values.get("removed_track_count")
            count = count if isinstance(count, int) else 0
            return ToolProgressPresentation(summary=f"歌单「{name}」· 已移除 {count} 首")
        if tool_name == "clear_music_queue":
            count = values.get("removed_count")
            count = count if isinstance(count, int) else 0
            stopped = "已停止播放 · " if values.get("stopped_current") else ""
            return ToolProgressPresentation(summary=f"{stopped}已移除 {count} 首")
        if tool_name == "load_music_playlist":
            name = self.scalar(values.get("playlist_name"))
            count = values.get("loaded_count")
            count = count if isinstance(count, int) else 0
            return ToolProgressPresentation(summary=f"歌单「{name}」· 已载入 {count} 首")
        if tool_name == "preview_netease_playlist":
            name = self.scalar(values.get("name"))
            declared = values.get("declared_track_count")
            visible = values.get("visible_track_count")
            declared = declared if isinstance(declared, int) else 0
            visible = visible if isinstance(visible, int) else 0
            suffix = " · 需要确认部分导入" if values.get("complete") is False else ""
            return ToolProgressPresentation(
                summary=f"歌单「{name}」· 可导入 {visible}/{declared} 首{suffix}"
            )
        if tool_name == "import_netease_playlist":
            playlist = values.get("playlist")
            name = self.scalar(playlist.get("name")) if isinstance(playlist, Mapping) else ""
            imported = values.get("imported_track_count")
            skipped = values.get("skipped_track_count")
            imported = imported if isinstance(imported, int) else 0
            skipped = skipped if isinstance(skipped, int) else 0
            suffix = f" · 跳过 {skipped} 首" if skipped else ""
            return ToolProgressPresentation(summary=f"歌单「{name}」· 已导入 {imported} 首{suffix}")
        if tool_name in {"skip_music", "pause_music", "resume_music"}:
            return ToolProgressPresentation(
                summary=("操作已生效" if values.get("applied") else "当前无需操作")
            )
        track = values.get("track")
        if isinstance(track, Mapping):
            title = self.scalar(track.get("title"))
            position = values.get("position")
            summary = self._join(
                self._track_source_name(track),
                f"歌曲「{title or '未知'}」",
            )
            if isinstance(position, int):
                summary += f" · 队列第 {position} 位"
            return ToolProgressPresentation(summary=summary)
        return ToolProgressPresentation(summary="调用完成")

    @classmethod
    def _track_source_name(cls, track: object) -> str:
        return cls._source_name(track.get("source")) if isinstance(track, Mapping) else ""

    @classmethod
    def _tracks_source_name(cls, tracks: object) -> str:
        if not isinstance(tracks, Sequence) or isinstance(tracks, str):
            return ""
        sources = {
            cls._track_source_name(track) for track in tracks if cls._track_source_name(track)
        }
        return next(iter(sources)) if len(sources) == 1 else ""

    @classmethod
    def _source_name(cls, value: object, *, target: object = None) -> str:
        normalized = cls.scalar(value, limit=32).casefold()
        if normalized and normalized != "auto":
            return cls._source_names.get(normalized, normalized)
        host = cls.host(target).casefold()
        if host.endswith("music.163.com"):
            return cls._source_names["netease"]
        if host.endswith(("youtube.com", "youtu.be")):
            return cls._source_names["youtube"]
        if host.endswith(("bilibili.com", "b23.tv")):
            return cls._source_names["bilibili"]
        return ""

    @staticmethod
    def _join(*values: str) -> str:
        return " · ".join(value for value in values if value)


class SkillProgressProjector(_ProjectionSupport):
    """Present Skill metadata and character counts without exposing loaded text."""

    def request(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        if tool_name == "list_agent_skill_library":
            return ToolProgressPresentation(subject="我的技能")
        if tool_name == "create_agent_skill":
            return ToolProgressPresentation(
                subject=self.scalar(arguments.get("display_name") or arguments.get("name"))
            )
        if tool_name == "manage_agent_skill_resource":
            return ToolProgressPresentation(subject=self.scalar(arguments.get("key")))
        if tool_name == "set_agent_skill_state":
            action = self.scalar(arguments.get("action"))
            return ToolProgressPresentation(
                summary="准备归档" if action == "archive" else "准备恢复"
            )
        if tool_name == "invite_agent_skill_share":
            return ToolProgressPresentation(subject="当前消息提及的用户")
        if tool_name == "respond_agent_skill_share":
            decision = self.scalar(arguments.get("decision"))
            return ToolProgressPresentation(
                summary="准备接受" if decision == "accept" else "准备拒绝"
            )
        if tool_name == "revoke_agent_skill_share":
            return ToolProgressPresentation(
                subject=(
                    "全部分享" if arguments.get("revoke_all") is True else "当前消息提及的用户"
                ),
                summary="准备撤销",
            )
        return ToolProgressPresentation()

    def result(
        self,
        tool_name: str,
        values: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        skill = values.get("skill")
        skill_values = skill if isinstance(skill, Mapping) else {}
        if tool_name == "list_agent_skill_library":
            owned = values.get("owned")
            shared = values.get("shared")
            pending = values.get("pending_invitations")
            owned_count = len(owned) if isinstance(owned, list | tuple) else 0
            shared_count = len(shared) if isinstance(shared, list | tuple) else 0
            pending_count = len(pending) if isinstance(pending, list | tuple) else 0
            return ToolProgressPresentation(
                summary=(
                    f"{owned_count} 个我的技能 · {shared_count} 个共享 · {pending_count} 个待接受"
                )
            )
        if tool_name == "inspect_agent_skill":
            subject = self.scalar(skill_values.get("display_name") or skill_values.get("name"))
            resources = values.get("resources")
            count = len(resources) if isinstance(resources, list | tuple) else 0
            return ToolProgressPresentation(
                subject=subject,
                summary=f"最新版本 · {count} 份资料",
            )
        if tool_name in {
            "create_agent_skill",
            "update_agent_skill",
            "manage_agent_skill_resource",
            "set_agent_skill_state",
        }:
            subject = self.scalar(skill_values.get("display_name") or skill_values.get("name"))
            resource_count = values.get("resource_count")
            suffix = (
                f" · {resource_count} 份资料"
                if isinstance(resource_count, int) and resource_count
                else ""
            )
            return ToolProgressPresentation(
                subject=subject,
                summary=f"已保存 · 下轮生效{suffix}",
            )
        if tool_name == "invite_agent_skill_share":
            subject = self.scalar(skill_values.get("display_name") or skill_values.get("name"))
            invitation_count = values.get("invitation_count")
            failures = values.get("notification_failures")
            count = invitation_count if isinstance(invitation_count, int) else 0
            failure_count = failures if isinstance(failures, int) else 0
            suffix = f" · {failure_count} 个通知失败" if failure_count else ""
            return ToolProgressPresentation(
                subject=subject,
                summary=f"已邀请 {count} 人{suffix}",
            )
        if tool_name == "respond_agent_skill_share":
            subject = self.scalar(skill_values.get("display_name") or skill_values.get("name"))
            accepted = values.get("status") == "accepted"
            return ToolProgressPresentation(
                subject=subject,
                summary="已接受 · 下轮可用" if accepted else "已拒绝",
            )
        if tool_name == "revoke_agent_skill_share":
            subject = self.scalar(skill_values.get("display_name") or skill_values.get("name"))
            revoked = values.get("revoked_count")
            failures = values.get("notification_failures")
            revoked_count = revoked if isinstance(revoked, int) else 0
            failure_count = failures if isinstance(failures, int) else 0
            suffix = f" · {failure_count} 个通知失败" if failure_count else ""
            return ToolProgressPresentation(
                subject=subject,
                summary=f"已撤销 {revoked_count} 项分享{suffix}",
            )
        repeated = values.get("already_loaded") is True
        characters = values.get("character_count")
        character_count = characters if isinstance(characters, int) else 0
        if tool_name == "load_agent_skill":
            subject = self.scalar(skill_values.get("display_name") or skill_values.get("name"))
            version = self.scalar(skill_values.get("version"), limit=32)
            summary = (
                f"v{version} · 已加载"
                if repeated
                else f"v{version} · {self._characters(character_count)}"
            )
            return ToolProgressPresentation(subject=subject, summary=summary)

        resource = values.get("resource")
        resource_values = resource if isinstance(resource, Mapping) else {}
        subject = self.scalar(resource_values.get("display_name") or resource_values.get("key"))
        summary = "已读取" if repeated else self._characters(character_count)
        return ToolProgressPresentation(subject=subject, summary=summary)

    @staticmethod
    def _characters(value: int) -> str:
        if value >= 1000:
            return f"{value / 1000:.1f}k 字"
        return f"{value} 字"


class GenericToolProgressProjector(_ProjectionSupport):
    """Conservative fallback that never exposes arbitrary result payloads."""

    def request(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        del tool_name
        for name in ("query", "url", "ref", "key", "emoji", "limit", "value"):
            if name not in arguments or self.sensitive(name):
                continue
            value = self.scalar(arguments[name])
            if value:
                return ToolProgressPresentation(subject=value)
        return ToolProgressPresentation()

    def result(
        self,
        tool_name: str,
        values: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        if tool_name == "react_to_message":
            emoji = self.scalar(values.get("emoji"))
            return ToolProgressPresentation(summary=f"已添加 {emoji}" if emoji else "回应已添加")
        if tool_name == "get_agent_status":
            remaining = values.get("remaining_tool_calls")
            if isinstance(remaining, int):
                return ToolProgressPresentation(summary=f"剩余 {remaining} 次工具调用")
        if tool_name == "get_channel_settings":
            return ToolProgressPresentation(
                summary=("频道聊天已开启" if values.get("chat_enabled") else "频道聊天未开启")
            )
        if isinstance(values.get("position"), int):
            return ToolProgressPresentation(summary=f"队列第 {values['position']} 位")
        return ToolProgressPresentation(summary="调用完成")


class ToolProgressCatalog:
    """Route known tool families to safe structured progress projectors."""

    _error_summaries = {
        "administrator_required": "需要管理员权限",
        "browser_action_failed": "网页操作失败",
        "browser_navigation_failed": "网页打开失败",
        "browser_stale_ref": "网页元素已失效，请刷新页面",
        "browser_timeout": "网页响应超时",
        "browser_unavailable": "浏览器暂不可用",
        "cancelled": "调用已取消",
        "duplicate_tool_call_in_progress": "相同调用仍在执行",
        "invalid_arguments": "调用参数不正确",
        "invalid_tool_output": "工具返回了无效结果",
        "invalid_web_search_query": "搜索内容不正确",
        "invalid_music_query": "歌曲搜索内容不正确",
        "invalid_music_reference": "歌曲来源或标识不正确",
        "music_authentication_required": "该音乐来源需要登录凭据",
        "music_catalog_unavailable": "音乐搜索服务暂不可用",
        "music_content_unsupported": "该链接不是受支持的单曲内容",
        "music_extraction_timeout": "音乐来源响应超时",
        "music_failed": "音乐工具执行失败",
        "music_geo_restricted": "该内容在当前地区不可用",
        "music_live_unsupported": "暂不支持直播内容",
        "music_no_audio_format": "没有找到可播放的音频格式",
        "music_not_found": "没有找到匹配的歌曲",
        "music_rate_limited": "音乐来源请求过于频繁",
        "music_source_disabled": "该音乐来源尚未启用",
        "music_source_unavailable": "该音乐来源暂不可用",
        "music_track_too_long": "歌曲超过允许的播放时长",
        "music_voice_busy": "语音频道正由语音对话占用",
        "music_area_required": "共享歌单只能在 area 内使用",
        "invalid_music_playlist_name": "歌单名称不正确",
        "music_playlist_exists": "当前 area 已有同名歌单",
        "music_playlist_not_found": "当前 area 没有这个歌单",
        "music_playlist_full": "歌单已满",
        "music_playlist_empty": "歌单还是空的",
        "music_playlist_unavailable": "共享歌单服务暂不可用",
        "music_playlist_failed": "歌单操作失败",
        "invalid_netease_playlist_reference": "网易云歌单 ID 或链接不正确",
        "netease_playlist_not_found": "没有找到这个网易云歌单",
        "netease_playlist_incomplete": "歌单内容不完整，需要确认后再部分导入",
        "netease_playlist_too_large": "歌单超过 area 歌单容量，需要确认后再部分导入",
        "music_playback_failed": "音乐播放操作失败",
        "music_queue_full": "播放队列已满",
        "music_voice_channel_required": "请先加入语音频道",
        "skill_activation_limit": "本轮加载的技能已达上限",
        "skill_catalog_unavailable": "技能目录暂不可用",
        "skill_context_limit": "本轮技能内容已达上限",
        "skill_load_failed": "技能加载失败",
        "skill_not_activated": "请先加载对应技能",
        "skill_not_available": "当前对话不能使用这个技能",
        "skill_not_found": "没有找到这个技能",
        "skill_revision_changed": "技能已被修改，请在下一轮重新加载",
        "skill_selector_ambiguous": "存在同名技能，请使用目录中的技能 ID",
        "skill_resource_limit": "本轮读取的技能资料已达上限",
        "skill_resource_not_found": "没有找到这份技能资料",
        "skill_conflict": "技能名称或资料位置发生冲突",
        "skill_instructions_too_long": "技能说明超过长度限制",
        "skill_library_limit": "个人技能库已满",
        "skill_library_unavailable": "技能库暂不可用",
        "skill_no_changes": "技能内容没有变化",
        "skill_not_owned": "只能修改自己创建的技能",
        "skill_resource_library_limit": "这项技能的资料数量已达上限",
        "skill_resource_too_long": "技能资料超过长度限制",
        "skill_revision_conflict": "内容已被修改，请重新查看后再编辑",
        "skill_unknown_required_tools": "技能引用了不存在的工具",
        "skill_archived": "这个技能当前已归档",
        "skill_share_target_required": "请在当前消息中 @ 要分享的用户",
        "skill_share_target_limit": "一次提及的分享对象过多",
        "skill_share_target_conflict": "全部撤销时不能同时指定用户",
        "skill_invitation_not_found": "没有找到属于你的这项技能邀请",
        "skill_invitation_answered": "这项技能邀请已经处理过",
        "skill_shared_library_limit": "已接受的共享技能已达上限",
        "skill_share_not_found": "没有找到这项技能分享",
        "invalid_agent_skill": "技能内容格式不正确",
        "invalid_agent_skill_resource": "技能资料格式不正确",
        "browser_failed": "浏览器操作失败",
        "tool_failed": "工具执行失败",
        "tool_not_enabled": "当前频道未启用此工具",
        "tool_not_registered": "工具当前不可用",
        "tool_timeout": "工具执行超时",
        "web_page_url_invalid": "网页地址不可访问",
        "web_search_rate_limited": "网页搜索请求过于频繁",
        "web_search_failed": "网页搜索失败",
        "web_search_timeout": "网页搜索超时",
        "web_search_unavailable": "网页搜索服务暂不可用",
    }

    def __init__(self) -> None:
        self._web_search = WebSearchProgressProjector()
        self._browser = BrowserProgressProjector()
        self._music = MusicProgressProjector()
        self._skills = SkillProgressProjector()
        self._generic = GenericToolProgressProjector()

    def request(self, tool_name: str, arguments: object) -> ToolProgressPresentation:
        """Project safe model arguments without retaining the raw payload."""
        values = self._as_mapping(arguments)
        if values is None:
            return ToolProgressPresentation()
        return self._projector(tool_name).request(tool_name, values)

    def result(
        self,
        tool_name: str,
        content: object,
        *,
        succeeded: bool,
    ) -> ToolProgressPresentation:
        """Project one safe model result envelope."""
        envelope = self._as_mapping(content)
        if not succeeded:
            code = str(envelope.get("error", "")) if envelope is not None else ""
            return ToolProgressPresentation(summary=self._error_summaries.get(code, "工具执行失败"))
        data = envelope.get("data") if envelope is not None else None
        values = data if isinstance(data, Mapping) else {}
        return self._projector(tool_name).result(tool_name, values)

    def _projector(self, tool_name: str) -> ToolProgressProjector:
        if tool_name == "search_web":
            return self._web_search
        if tool_name == "read_web_page" or tool_name.startswith("browser_"):
            return self._browser
        if (
            tool_name.endswith("_music")
            or "music_" in tool_name
            or tool_name in {"preview_netease_playlist", "import_netease_playlist"}
        ):
            return self._music
        if tool_name in {
            "load_agent_skill",
            "read_agent_skill_resource",
            "list_agent_skill_library",
            "inspect_agent_skill",
            "create_agent_skill",
            "update_agent_skill",
            "manage_agent_skill_resource",
            "set_agent_skill_state",
            "invite_agent_skill_share",
            "respond_agent_skill_share",
            "revoke_agent_skill_share",
        }:
            return self._skills
        return self._generic

    @staticmethod
    def _as_mapping(value: object) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            return value
        if not isinstance(value, str):
            return None
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, Mapping) else None


# Temporary import compatibility for integrations outside this repository.
ToolProgressFormatter = ToolProgressCatalog
