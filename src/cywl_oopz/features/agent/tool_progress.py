"""Build short, display-safe details for Agent tool progress."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_WHITESPACE = re.compile(r"\s+")


class ToolProgressFormatter:
    """Summarize model-visible tool envelopes without exposing raw payloads."""

    max_detail_characters = 140
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
    _argument_labels = {
        "query": "查询",
        "url": "网址",
        "time_range": "时段",
        "ref": "元素",
        "text": "内容",
        "key": "按键",
        "emoji": "表情",
        "limit": "数量",
        "value": "参数",
    }
    _error_labels = {
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

    def request(self, tool_name: str, arguments: object) -> str:
        """Return a bounded summary of explicitly safe input fields."""
        values = self._as_mapping(arguments)
        if values is None:
            return ""
        ordered_names = (
            "query",
            "url",
            "time_range",
            "ref",
            "text",
            "key",
            "emoji",
            "limit",
            "value",
        )
        names = list(ordered_names) + sorted(set(values) - set(ordered_names))
        details: list[str] = []
        for name in names:
            if name not in values or self._sensitive(name):
                continue
            rendered = self._scalar(values[name])
            if not rendered:
                continue
            label = self._argument_labels.get(name, name.replace("_", " "))
            if name in {"query", "text"}:
                rendered = f"「{rendered}」"
            details.append(f"{label}：{rendered}")
            if len(details) == 2:
                break
        return self._bounded(" · ".join(details))

    def result(self, tool_name: str, content: object, *, succeeded: bool) -> str:
        """Return a human-readable success/error summary from the safe envelope."""
        envelope = self._as_mapping(content)
        if not succeeded:
            code = str(envelope.get("error", "")) if envelope is not None else ""
            return self._bounded(f"错误：{self._error_labels.get(code, '工具执行失败')}")
        data = envelope.get("data") if envelope is not None else None
        values = data if isinstance(data, Mapping) else {}

        if tool_name == "search_web":
            results = values.get("results")
            count = (
                len(results)
                if isinstance(results, Sequence) and not isinstance(results, str)
                else 0
            )
            return f"找到 {count} 条结果"
        if tool_name == "read_web_page":
            return self._page_summary(values, prefix="已读取")
        if tool_name.startswith("browser_"):
            if tool_name == "browser_close":
                return "浏览器已关闭" if values.get("closed") else "没有活动的浏览器"
            if tool_name == "browser_fill":
                return "输入框已填写" if values.get("applied") else "输入框未改变"
            return self._page_summary(values, prefix="当前页面")
        if tool_name == "search_music_catalog":
            tracks = values.get("tracks")
            count = (
                len(tracks) if isinstance(tracks, Sequence) and not isinstance(tracks, str) else 0
            )
            return f"找到 {count} 首歌曲"
        if tool_name == "get_music_queue":
            upcoming = values.get("upcoming")
            count = (
                len(upcoming)
                if isinstance(upcoming, Sequence) and not isinstance(upcoming, str)
                else 0
            )
            current = values.get("current")
            return f"正在播放 · 后续 {count} 首" if current else f"当前未播放 · 后续 {count} 首"
        if tool_name in {"skip_music", "pause_music", "resume_music"}:
            return "操作已生效" if values.get("applied") else "当前无需操作"
        if tool_name == "react_to_message":
            emoji = self._scalar(values.get("emoji"))
            return self._bounded(f"已添加 {emoji}") if emoji else "回应已添加"
        if tool_name == "get_agent_status":
            remaining = values.get("remaining_tool_calls")
            if isinstance(remaining, int):
                return f"剩余 {remaining} 次工具调用"
        if tool_name == "get_channel_settings":
            return "频道聊天已开启" if values.get("chat_enabled") else "频道聊天未开启"
        track = values.get("track")
        if isinstance(track, Mapping):
            title = self._scalar(track.get("title"))
            position = values.get("position")
            suffix = f" · 队列第 {position} 位" if isinstance(position, int) else ""
            return self._bounded(f"歌曲「{title or '未知'}」{suffix}")
        if isinstance(values.get("status"), str):
            return self._bounded(f"状态：{values['status']}")
        if isinstance(values.get("position"), int):
            return f"队列第 {values['position']} 位"
        return "调用完成"

    def _page_summary(self, values: Mapping[str, Any], *, prefix: str) -> str:
        title = self._scalar(values.get("title"))
        url = self._scalar(values.get("url"))
        target = f"「{title}」" if title else url
        suffix = "（内容已截断）" if values.get("truncated") is True else ""
        return self._bounded(f"{prefix}{target or '网页'}{suffix}")

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

    @classmethod
    def _sensitive(cls, name: str) -> bool:
        normalized = name.casefold()
        return any(part in normalized for part in cls._sensitive_name_parts)

    @classmethod
    def _scalar(cls, value: object) -> str:
        if value is None or isinstance(value, Mapping | Sequence) and not isinstance(value, str):
            return ""
        rendered = _WHITESPACE.sub(" ", str(value)).strip()
        if not rendered:
            return ""
        return rendered[:72] + ("…" if len(rendered) > 72 else "")

    def _bounded(self, value: str) -> str:
        normalized = _WHITESPACE.sub(" ", value).strip()
        if len(normalized) <= self.max_detail_characters:
            return normalized
        return normalized[: self.max_detail_characters - 1] + "…"
