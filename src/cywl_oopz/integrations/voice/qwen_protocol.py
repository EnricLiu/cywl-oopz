"""Pure Qwen Omni Realtime wire configuration, encoding, and event mapping."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from cywl_oopz.features.voice.audio import PROVIDER_INPUT_FORMAT, PROVIDER_OUTPUT_FORMAT
from cywl_oopz.features.voice.errors import (
    VoiceProviderAudioFormatError,
    VoiceProviderConfigurationError,
    VoiceProviderProtocolError,
)
from cywl_oopz.features.voice.events import (
    VoiceAssistantAudio,
    VoiceModelEvent,
    VoiceProviderFailed,
    VoiceProviderErrorEvent,
    VoiceResponseCancelled,
    VoiceResponseCompleted,
    VoiceResponseStarted,
    VoiceSessionFinished,
    VoiceSessionReady,
    VoiceToolCall,
    VoiceTranscriptFinal,
    VoiceUserSpeechStarted,
    VoiceUserSpeechStopped,
)
from cywl_oopz.features.voice.models import PcmChunk
from cywl_oopz.features.voice.settings import (
    VoiceProviderProtocol,
    VoiceStartConfiguration,
)

MAX_SERVER_EVENT_BYTES = 2 * 1024 * 1024


class QwenTurnDetectionMode(StrEnum):
    SERVER_VAD = "server_vad"
    SEMANTIC_VAD = "semantic_vad"


@dataclass(frozen=True, slots=True)
class QwenOmniConfig:
    endpoint: str
    api_key: str = field(repr=False)
    model: str = ""
    voice: str = "Tina"
    turn_detection: QwenTurnDetectionMode = QwenTurnDetectionMode.SEMANTIC_VAD
    vad_threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 700
    transcription_model: str = "qwen3-asr-flash-realtime"
    connect_timeout_seconds: float = 10.0

    @classmethod
    def from_start_configuration(
        cls,
        configuration: VoiceStartConfiguration,
    ) -> QwenOmniConfig:
        provider = configuration.provider
        model = configuration.model
        if provider.protocol is not VoiceProviderProtocol.QWEN_OMNI_REALTIME_WS:
            raise VoiceProviderConfigurationError("Voice model is not a Qwen Omni provider")
        endpoint = provider.endpoint.strip()
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
            raise VoiceProviderConfigurationError("Qwen endpoint must be a WebSocket URL")
        api_key = provider.credentials.get("api_key", "")
        if not isinstance(api_key, str) or not api_key.strip():
            raise VoiceProviderConfigurationError("Qwen api_key is missing")
        audio = model.audio_config
        _require_rate(audio, "input_sample_rate", PROVIDER_INPUT_FORMAT.sample_rate)
        _require_rate(audio, "output_sample_rate", PROVIDER_OUTPUT_FORMAT.sample_rate)
        default_vad = (
            "semantic_vad" if model.remote_model_name.startswith("qwen3.5-") else "server_vad"
        )
        try:
            turn_detection = QwenTurnDetectionMode(_string(audio, "turn_detection", default_vad))
        except ValueError as exc:
            raise VoiceProviderConfigurationError("Unsupported Qwen turn detection mode") from exc
        threshold = _number(audio, "vad_threshold", 0.5)
        if not 0 <= threshold <= 1:
            raise VoiceProviderConfigurationError("Qwen VAD threshold must be between 0 and 1")
        prefix_padding_ms = _integer(audio, "prefix_padding_ms", 300, minimum=0, maximum=2_000)
        silence_duration_ms = _integer(
            audio,
            "silence_duration_ms",
            700,
            minimum=100,
            maximum=5_000,
        )
        connect_timeout = _number(model.limits, "connect_timeout_seconds", 10.0)
        if connect_timeout <= 0 or connect_timeout > 60:
            raise VoiceProviderConfigurationError("Qwen connect timeout is outside 0-60 seconds")
        voice = configuration.voice_id or _string(provider.config, "default_voice", "Tina")
        if not voice or len(voice) > 128:
            raise VoiceProviderConfigurationError("Qwen voice identifier is invalid")
        return cls(
            endpoint=endpoint,
            api_key=api_key.strip(),
            model=model.remote_model_name,
            voice=voice,
            turn_detection=turn_detection,
            vad_threshold=threshold,
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
            transcription_model=_string(
                audio,
                "transcription_model",
                "qwen3-asr-flash-realtime",
            ),
            connect_timeout_seconds=connect_timeout,
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
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        session: dict[str, Any] = {
            "modalities": ["text", "audio"],
            "voice": self.voice,
            "instructions": instructions,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "input_audio_transcription": {"model": self.transcription_model},
            "turn_detection": {
                "type": self.turn_detection.value,
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.prefix_padding_ms,
                "silence_duration_ms": self.silence_duration_ms,
            },
        }
        if tools:
            session["tools"] = [dict(tool) for tool in tools]
        return {
            "event_id": event_id,
            "type": "session.update",
            "session": session,
        }


def encode_client_event(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


def audio_append_event(pcm: bytes, event_id: str) -> dict[str, str]:
    if not pcm or len(pcm) % PROVIDER_INPUT_FORMAT.frame_width_bytes:
        raise VoiceProviderAudioFormatError("Qwen input PCM must contain aligned samples")
    return {
        "event_id": event_id,
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode("ascii"),
    }


def function_call_output_event(
    call_id: str,
    output: Mapping[str, object],
    event_id: str,
) -> dict[str, object]:
    if not call_id.strip() or len(call_id) > 256:
        raise VoiceProviderProtocolError("Qwen tool call identifier is invalid")
    encoded = json.dumps(dict(output), ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 16_384:
        raise VoiceProviderProtocolError("Qwen tool output exceeded the maximum size")
    return {
        "event_id": event_id,
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": encoded,
        },
    }


def response_create_event(event_id: str) -> dict[str, str]:
    return {"event_id": event_id, "type": "response.create"}


def internal_context_event(item_id: str, text: str, event_id: str) -> dict[str, object]:
    if not item_id.strip() or len(item_id) > 128:
        raise VoiceProviderProtocolError("Qwen internal item identifier is invalid")
    if not text.strip() or len(text.encode("utf-8")) > 16_384:
        raise VoiceProviderProtocolError("Qwen internal context text is invalid")
    return {
        "event_id": event_id,
        "type": "conversation.item.create",
        "item": {
            "id": item_id,
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def parse_server_event(raw: str | bytes) -> VoiceModelEvent | None:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VoiceProviderProtocolError("Qwen sent a non-UTF-8 event") from exc
    if len(raw.encode("utf-8")) > MAX_SERVER_EVENT_BYTES:
        raise VoiceProviderProtocolError("Qwen event exceeded the maximum size")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VoiceProviderProtocolError("Qwen sent malformed JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        raise VoiceProviderProtocolError("Qwen event type is missing")
    event_type = payload["type"]
    if event_type == "session.updated":
        return VoiceSessionReady()
    if event_type == "session.finished":
        return VoiceSessionFinished()
    if event_type == "input_audio_buffer.speech_started":
        return VoiceUserSpeechStarted()
    if event_type == "input_audio_buffer.speech_stopped":
        return VoiceUserSpeechStopped()
    if event_type == "response.created":
        response = payload.get("response")
        if not isinstance(response, dict):
            raise VoiceProviderProtocolError("Qwen response.created omitted the response")
        return VoiceResponseStarted(_required_identifier(response, "id", "response"))
    if event_type == "conversation.item.input_audio_transcription.completed":
        return _transcript(payload, "user")
    if event_type == "response.audio_transcript.done":
        return _transcript(payload, "assistant")
    if event_type == "response.audio.delta":
        return _audio_delta(payload)
    if event_type == "response.function_call_arguments.done":
        return _tool_call(payload)
    if event_type == "response.done":
        response = payload.get("response")
        if not isinstance(response, dict):
            raise VoiceProviderProtocolError("Qwen response.done omitted the response")
        status = response.get("status")
        if status == "failed":
            return VoiceProviderFailed("response_failed", retryable=False)
        response_id = _required_identifier(response, "id", "response")
        usage = _usage(response.get("usage"))
        if status == "cancelled":
            return VoiceResponseCancelled(response_id, usage)
        if status in {"completed", "incomplete"}:
            return VoiceResponseCompleted(response_id, usage)
        raise VoiceProviderProtocolError("Qwen response.done used an unknown status")
    if event_type == "error":
        error = payload.get("error")
        error = error if isinstance(error, dict) else {}
        typ = str(error.get("type", "anonymous_error_type"))
        code = str(error.get("code", "anonymous_error_code"))
        msg = str(error.get("message", "Qwen sent an unknown error"))
        param = str(error.get("param", ""))
        normalized = code.casefold()
        retryable = any(
            marker in normalized
            for marker in ("rate", "limit", "timeout", "overload", "server", "unavailable")
        )
        return VoiceProviderErrorEvent(typ, code, msg, param, retryable)
    return None


def _tool_call(payload: dict[str, Any]) -> VoiceToolCall:
    call_id = _required_identifier(payload, "call_id", "tool call")
    name = _required_identifier(payload, "name", "tool")
    if len(name) > 128:
        raise VoiceProviderProtocolError("Qwen tool name is invalid")
    encoded = payload.get("arguments")
    if not isinstance(encoded, str) or len(encoded.encode("utf-8")) > 16_384:
        raise VoiceProviderProtocolError("Qwen tool arguments are invalid")
    try:
        arguments = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise VoiceProviderProtocolError("Qwen tool arguments are malformed") from exc
    if not isinstance(arguments, dict):
        raise VoiceProviderProtocolError("Qwen tool arguments must be an object")
    return VoiceToolCall(call_id, name, arguments)


def _transcript(payload: dict[str, Any], role: str) -> VoiceTranscriptFinal | None:
    transcript = payload.get("transcript", "")
    if not isinstance(transcript, str) or not transcript.strip():
        return None
    item_id = payload.get("item_id", "")
    response_id = payload.get("response_id", "") if role == "assistant" else ""
    return VoiceTranscriptFinal(
        role,
        transcript.strip(),
        str(item_id or "")[:256],
        str(response_id or "")[:256],
    )


def _audio_delta(payload: dict[str, Any]) -> VoiceAssistantAudio:
    encoded = payload.get("delta")
    if not isinstance(encoded, str) or not encoded:
        raise VoiceProviderAudioFormatError("Qwen audio delta is empty")
    try:
        pcm = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VoiceProviderAudioFormatError("Qwen audio delta is not valid base64") from exc
    if not pcm or len(pcm) % PROVIDER_OUTPUT_FORMAT.frame_width_bytes:
        raise VoiceProviderAudioFormatError("Qwen output PCM is not sample aligned")
    samples = len(pcm) // PROVIDER_OUTPUT_FORMAT.frame_width_bytes
    duration_ms = round(samples * 1000 / PROVIDER_OUTPUT_FORMAT.sample_rate)
    if duration_ms <= 0:
        raise VoiceProviderAudioFormatError("Qwen output PCM delta is too short")
    try:
        chunk = PcmChunk(pcm, PROVIDER_OUTPUT_FORMAT, duration_ms, generation=0)
    except ValueError as exc:
        raise VoiceProviderAudioFormatError("Qwen output PCM duration is invalid") from exc
    return VoiceAssistantAudio(
        chunk,
        _required_identifier(payload, "response_id", "response"),
        _required_identifier(payload, "item_id", "item"),
    )


def response_cancel_event(event_id: str) -> dict[str, str]:
    return {"event_id": event_id, "type": "response.cancel"}


def _usage(value: object) -> dict[str, int | float]:
    if not isinstance(value, dict):
        return {}
    usage: dict[str, int | float] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, int | float) and not isinstance(item, bool) and item >= 0:
            usage[key] = item
    return usage


def _required_identifier(payload: dict[str, Any], key: str, kind: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise VoiceProviderProtocolError(f"Qwen {kind} identifier is invalid")
    return value


def _require_rate(values: Any, key: str, expected: int) -> None:
    value = values.get(key, expected)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise VoiceProviderConfigurationError(f"Qwen {key} must be {expected}")


def _string(values: Any, key: str, default: str) -> str:
    value = values.get(key, default)
    if not isinstance(value, str):
        raise VoiceProviderConfigurationError(f"Qwen {key} must be a string")
    return value.strip()


def _number(values: Any, key: str, default: float) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise VoiceProviderConfigurationError(f"Qwen {key} must be numeric")
    return float(value)


def _integer(
    values: Any,
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise VoiceProviderConfigurationError(f"Qwen {key} must be between {minimum} and {maximum}")
    return value
