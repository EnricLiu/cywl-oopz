from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import pytest

from cywl_oopz.features.voice.errors import (
    VoiceProviderAudioFormatError,
    VoiceProviderConfigurationError,
    VoiceProviderProtocolError,
)
from cywl_oopz.features.voice.events import (
    VoiceAssistantAudio,
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
from cywl_oopz.features.voice.models import VoiceChannelKey
from cywl_oopz.features.voice.settings import (
    VoiceChannelConfiguration,
    VoiceDuplexMode,
    VoiceModelConfiguration,
    VoiceModelMode,
    VoiceProviderConfiguration,
    VoiceProviderProtocol,
    VoiceStartConfiguration,
)
from cywl_oopz.integrations.voice.qwen_protocol import (
    QwenOmniConfig,
    QwenTurnDetectionMode,
    audio_append_event,
    encode_client_event,
    function_call_output_event,
    parse_server_event,
    response_create_event,
)


def configuration(**audio_overrides) -> VoiceStartConfiguration:
    provider_id = uuid4()
    return VoiceStartConfiguration(
        provider=VoiceProviderConfiguration(
            provider_id,
            "qwen",
            "Qwen",
            VoiceProviderProtocol.QWEN_OMNI_REALTIME_WS,
            "wss://workspace.example/api-ws/v1/realtime?region=cn",
            MappingProxyType({"api_key": "development-secret"}),
            MappingProxyType({"default_voice": "Ethan"}),
        ),
        model=VoiceModelConfiguration(
            uuid4(),
            provider_id,
            "omni",
            "qwen3.5-omni-flash-realtime",
            "Qwen Omni",
            VoiceModelMode.NATIVE_REALTIME,
            MappingProxyType({}),
            MappingProxyType(audio_overrides),
            MappingProxyType({}),
            MappingProxyType({"connect_timeout_seconds": 4}),
        ),
        channel=VoiceChannelConfiguration(
            VoiceChannelKey("area", "voice"),
            "voice_readonly_v1",
            300,
        ),
        voice_id="Cherry",
        duplex_mode=VoiceDuplexMode.FULL,
        delegated_agent_model_id=None,
    )


def test_qwen_config_parses_pinned_catalog_without_exposing_credentials() -> None:
    config = QwenOmniConfig.from_start_configuration(configuration())

    assert config.url.endswith("?region=cn&model=qwen3.5-omni-flash-realtime")
    assert config.voice == "Cherry"
    assert config.turn_detection is QwenTurnDetectionMode.SEMANTIC_VAD
    assert "development-secret" not in repr(config)
    update = config.session_update("short prompt", "event-1")
    assert update["session"]["input_audio_format"] == "pcm"
    assert update["session"]["output_audio_format"] == "pcm"
    assert update["session"]["turn_detection"]["type"] == "semantic_vad"
    assert "development-secret" not in encode_client_event(update)


@pytest.mark.parametrize(
    "overrides",
    (
        {"input_sample_rate": 48_000},
        {"output_sample_rate": 16_000},
        {"turn_detection": "unknown"},
        {"vad_threshold": 2},
        {"silence_duration_ms": 10},
    ),
)
def test_qwen_config_rejects_incompatible_audio_contract(overrides) -> None:
    with pytest.raises(VoiceProviderConfigurationError):
        QwenOmniConfig.from_start_configuration(configuration(**overrides))


def test_qwen_recorded_fixture_maps_only_curated_domain_events() -> None:
    fixture = Path("tests/fixtures/qwen_omni_realtime_contract.json")
    payloads = json.loads(fixture.read_text(encoding="utf-8"))

    events = [parse_server_event(json.dumps(payload)) for payload in payloads]

    assert events[0] is None
    assert isinstance(events[1], VoiceSessionReady)
    assert isinstance(events[2], VoiceUserSpeechStarted)
    assert isinstance(events[3], VoiceUserSpeechStopped)
    assert events[4] == VoiceTranscriptFinal("user", "你好", "item-user")
    assert events[5] == VoiceResponseStarted("response-1")
    assert events[6] == VoiceTranscriptFinal("assistant", "你好呀", "item-assistant", "response-1")
    assert isinstance(events[7], VoiceAssistantAudio)
    assert events[7].chunk.duration_ms == 1
    assert events[7].response_id == "response-1"
    assert events[7].provider_item_id == "item-assistant"
    assert isinstance(events[8], VoiceResponseCompleted)
    assert events[8].response_id == "response-1"
    assert events[8].usage == {
        "input_tokens": 12,
        "output_tokens": 5,
        "total_tokens": 17,
    }
    assert events[9] == VoiceProviderErrorEvent(
        "anonymous_error_type",
        "rate_limit_exceeded",
        "fixture-only",
        "",
        True,
    )
    assert events[10] is None
    assert events[11] == VoiceResponseStarted("response-2")
    assert events[12] == VoiceResponseCancelled("response-2", {})
    assert isinstance(events[13], VoiceSessionFinished)


def test_qwen_codec_rejects_malformed_json_base64_and_input_alignment() -> None:
    with pytest.raises(VoiceProviderProtocolError):
        parse_server_event("not-json")
    with pytest.raises(VoiceProviderAudioFormatError):
        parse_server_event('{"type":"response.audio.delta","delta":"%%%"}')
    with pytest.raises(VoiceProviderAudioFormatError):
        audio_append_event(b"\x00", "event-1")


def test_qwen_function_call_contract_is_bounded_and_typed() -> None:
    event = parse_server_event(
        json.dumps(
            {
                "type": "response.function_call_arguments.done",
                "call_id": "call-1",
                "name": "delegate_agent_task",
                "arguments": '{"objective":"查一下演出"}',
            }
        )
    )

    assert event == VoiceToolCall(
        "call-1",
        "delegate_agent_task",
        {"objective": "查一下演出"},
    )
    output = function_call_output_event("call-1", {"ok": True, "task": "T1"}, "event-2")
    assert output["type"] == "conversation.item.create"
    assert output["item"]["type"] == "function_call_output"
    assert json.loads(output["item"]["output"]) == {"ok": True, "task": "T1"}
    assert response_create_event("event-3")["type"] == "response.create"
    with pytest.raises(VoiceProviderProtocolError):
        parse_server_event(
            '{"type":"response.function_call_arguments.done",'
            '"call_id":"c","name":"delegate_agent_task","arguments":"[]"}'
        )
    with pytest.raises(VoiceProviderProtocolError):
        parse_server_event(
            json.dumps(
                {
                    "type": "response.function_call_arguments.done",
                    "call_id": "c",
                    "name": "x" * 129,
                    "arguments": "{}",
                }
            )
        )
