"""Capability-gated strategy selection for delegated task notifications."""

from __future__ import annotations

from enum import StrEnum

from .models import VoiceProviderCapabilities


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
