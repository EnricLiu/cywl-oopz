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

    def request(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolProgressPresentation:
        del tool_name
        return ToolProgressPresentation(subject=self.scalar(arguments.get("query")))

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
            return ToolProgressPresentation(summary=f"找到 {count} 首歌曲")
        if tool_name == "get_music_queue":
            upcoming = values.get("upcoming")
            count = (
                len(upcoming)
                if isinstance(upcoming, Sequence) and not isinstance(upcoming, str)
                else 0
            )
            current = values.get("current")
            summary = f"正在播放 · 后续 {count} 首" if current else f"当前未播放 · 后续 {count} 首"
            return ToolProgressPresentation(summary=summary)
        if tool_name in {"skip_music", "pause_music", "resume_music"}:
            return ToolProgressPresentation(
                summary=("操作已生效" if values.get("applied") else "当前无需操作")
            )
        track = values.get("track")
        if isinstance(track, Mapping):
            title = self.scalar(track.get("title"))
            position = values.get("position")
            summary = f"歌曲「{title or '未知'}」"
            if isinstance(position, int):
                summary += f" · 队列第 {position} 位"
            return ToolProgressPresentation(summary=summary)
        return ToolProgressPresentation(summary="调用完成")


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
        "music_catalog_unavailable": "音乐搜索服务暂不可用",
        "music_failed": "音乐工具执行失败",
        "music_not_found": "没有找到匹配的歌曲",
        "music_playback_failed": "音乐播放操作失败",
        "music_queue_full": "播放队列已满",
        "music_voice_channel_required": "请先加入语音频道",
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
        if tool_name.endswith("_music") or "music_" in tool_name:
            return self._music
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
