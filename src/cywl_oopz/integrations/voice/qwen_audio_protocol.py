"""Qwen-Audio Realtime configuration over the shared Qwen wire event mapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from cywl_oopz.features.voice.audio import PROVIDER_INPUT_FORMAT, PROVIDER_OUTPUT_FORMAT
from cywl_oopz.features.voice.errors import VoiceProviderConfigurationError
from cywl_oopz.features.voice.settings import VoiceProviderProtocol, VoiceStartConfiguration


class QwenAudioTurnDetectionMode(StrEnum):
    SERVER_VAD = "server_vad"
    SMART_TURN = "smart_turn"


@dataclass(frozen=True, slots=True)
class QwenAudioConfig:
    endpoint: str
    api_key: str = field(repr=False)
    model: str = ""
    voice: str = "longanqian"
    turn_detection: QwenAudioTurnDetectionMode = QwenAudioTurnDetectionMode.SMART_TURN
    vad_threshold: float = 0.5
    silence_duration_ms: int = 800
    max_history_turns: int = 20
    connect_timeout_seconds: float = 10.0

    @classmethod
    def from_start_configuration(
        cls,
        configuration: VoiceStartConfiguration,
    ) -> QwenAudioConfig:
        provider = configuration.provider
        model = configuration.model
        if provider.protocol is not VoiceProviderProtocol.QWEN_AUDIO_REALTIME_WS:
            raise VoiceProviderConfigurationError("Voice model is not a Qwen Audio provider")
        endpoint = provider.endpoint.strip()
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise VoiceProviderConfigurationError("Qwen Audio endpoint must be a WebSocket URL")
        api_key = provider.credentials.get("api_key", "")
        if not isinstance(api_key, str) or not api_key.strip():
            raise VoiceProviderConfigurationError("Qwen Audio api_key is missing")
        _require_rate(model.audio_config, "input_sample_rate", PROVIDER_INPUT_FORMAT.sample_rate)
        _require_rate(model.audio_config, "output_sample_rate", PROVIDER_OUTPUT_FORMAT.sample_rate)
        try:
            turn_detection = QwenAudioTurnDetectionMode(
                _string(model.audio_config, "turn_detection", "smart_turn")
            )
        except ValueError as exc:
            raise VoiceProviderConfigurationError(
                "Unsupported Qwen Audio turn detection mode"
            ) from exc
        threshold = _number(model.audio_config, "vad_threshold", 0.5)
        if not 0 <= threshold <= 1:
            raise VoiceProviderConfigurationError(
                "Qwen Audio VAD threshold must be between 0 and 1"
            )
        silence_duration_ms = _integer(
            model.audio_config,
            "silence_duration_ms",
            800,
            minimum=100,
            maximum=5_000,
        )
        max_history_turns = _integer(
            model.limits,
            "max_history_turns",
            20,
            minimum=1,
            maximum=50,
        )
        connect_timeout = _number(model.limits, "connect_timeout_seconds", 10.0)
        if connect_timeout <= 0 or connect_timeout > 60:
            raise VoiceProviderConfigurationError(
                "Qwen Audio connect timeout is outside 0-60 seconds"
            )
        voice = configuration.voice_id or _string(
            provider.config,
            "default_voice",
            "longanqian",
        )
        if not voice or len(voice) > 128:
            raise VoiceProviderConfigurationError("Qwen Audio voice identifier is invalid")
        return cls(
            endpoint,
            api_key.strip(),
            model.remote_model_name,
            voice,
            turn_detection,
            threshold,
            silence_duration_ms,
            max_history_turns,
            connect_timeout,
        )

    @property
    def url(self) -> str:
        parsed = urlsplit(self.endpoint)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["model"] = self.model
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
        )

    def session_update(
        self,
        instructions: str,
        event_id: str,
        tools: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, object]:
        session: dict[str, object] = {
            "modalities": ["text", "audio"],
            "voice": self.voice,
            "instructions": instructions,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "max_history_turns": self.max_history_turns,
            "turn_detection": {
                "type": self.turn_detection.value,
                "threshold": self.vad_threshold,
                "silence_duration_ms": self.silence_duration_ms,
            },
        }
        if tools:
            session["tools"] = [_audio_tool_schema(tool) for tool in tools]
        return {
            "event_id": event_id,
            "type": "session.update",
            "session": session,
        }


def _audio_tool_schema(tool: Mapping[str, object]) -> dict[str, object]:
    name = tool.get("name")
    if tool.get("type") != "function" or not isinstance(name, str) or not name:
        raise VoiceProviderConfigurationError("Qwen Audio tool schema is invalid")
    function = {key: value for key, value in tool.items() if key not in {"type", "name"}}
    function["name"] = name
    return {"type": "function", "function": function}


def _require_rate(values: Mapping[str, Any], key: str, expected: int) -> None:
    value = values.get(key, expected)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise VoiceProviderConfigurationError(f"Qwen Audio {key} must be {expected}")


def _string(values: Mapping[str, Any], key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str):
        raise VoiceProviderConfigurationError(f"Qwen Audio {key} must be a string")
    return value.strip()


def _number(values: Mapping[str, Any], key: str, default: float) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VoiceProviderConfigurationError(f"Qwen Audio {key} must be numeric")
    return float(value)


def _integer(
    values: Mapping[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise VoiceProviderConfigurationError(
            f"Qwen Audio {key} must be between {minimum} and {maximum}"
        )
    return value
