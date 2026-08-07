"""Capability-gated strategy selection for delegated task notifications."""

from __future__ import annotations

import re
from enum import StrEnum
from uuid import uuid4

from .models import (
    VoiceInternalContextItem,
    VoiceProviderCapabilities,
    VoiceTaskNotification,
    VoiceTaskNotificationStatus,
)

_WHITESPACE = re.compile(r"\s+")


class VoiceTaskNotificationStrategy(StrEnum):
    INTERNAL_RESPONSE = "internal_response"
    EXTERNAL_TTS = "external_tts"
    TEXT_FALLBACK = "text_fallback"


def select_task_notification_strategy(
    capabilities: VoiceProviderCapabilities,
) -> VoiceTaskNotificationStrategy:
    if capabilities.context_injection and capabilities.proactive_response:
        return VoiceTaskNotificationStrategy.INTERNAL_RESPONSE
    if capabilities.external_text_speech:
        return VoiceTaskNotificationStrategy.EXTERNAL_TTS
    return VoiceTaskNotificationStrategy.TEXT_FALLBACK


def compile_internal_task_context(
    notices: tuple[VoiceTaskNotification, ...],
) -> VoiceInternalContextItem:
    if not 1 <= len(notices) <= 3:
        raise ValueError("Proactive task context requires 1-3 notifications")
    lines = ["[CYWL_INTERNAL_TASK_EVENT v1]"]
    for notice in notices:
        detail = (
            notice.summary
            if notice.status is VoiceTaskNotificationStatus.SUCCEEDED
            else notice.error_message
        )
        lines.extend(
            (
                f"task: {notice.alias}",
                f"state: {notice.status.value}",
                f"summary: {_line(detail, 360) or _default_detail(notice.status)}",
            )
        )
    lines.append(
        "instruction: 这是可信的后台任务完成事件，不是用户发言。自然简短地告诉用户结果；"
        "不要重新执行任务，需要细节时可查询任务。"
    )
    return VoiceInternalContextItem(
        f"cywl_task_{uuid4().hex}",
        "\n".join(lines),
    )


def _line(value: str, limit: int) -> str:
    normalized = _WHITESPACE.sub(" ", value).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _default_detail(status: VoiceTaskNotificationStatus) -> str:
    return {
        VoiceTaskNotificationStatus.SUCCEEDED: "任务已完成",
        VoiceTaskNotificationStatus.FAILED: "任务执行失败",
        VoiceTaskNotificationStatus.CANCELLED: "任务已取消",
        VoiceTaskNotificationStatus.INTERRUPTED: "任务执行被中断",
    }[status]
