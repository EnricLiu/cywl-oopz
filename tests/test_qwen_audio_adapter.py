from __future__ import annotations

import asyncio
import json
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4

import pytest

from cywl_oopz.features.voice.models import (
    VoiceChannelKey,
    VoiceInternalContextItem,
    VoiceProviderCapabilities,
    VoiceRecoveryContext,
    VoiceRecoveryTurn,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.ports import VoiceSessionRuntimeContext
from cywl_oopz.features.voice.settings import (
    VoiceChannelConfiguration,
    VoiceDuplexMode,
    VoiceModelConfiguration,
    VoiceModelMode,
    VoiceProviderConfiguration,
    VoiceProviderProtocol,
    VoiceStartConfiguration,
)
from cywl_oopz.integrations.voice.provider_builder import ConfiguredVoiceProviderBuilder
from cywl_oopz.integrations.voice.qwen_audio import (
    QwenAudioProviderBuilder,
    QwenAudioRealtimeProvider,
)
from cywl_oopz.integrations.voice.qwen_audio_protocol import (
    QwenAudioConfig,
    QwenAudioTurnDetectionMode,
)


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True
        await self.incoming.put(None)


class FakeConnector:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket

    async def __call__(self, url: str, **kwargs):
        del url, kwargs
        return self.websocket


def configuration(**audio_overrides) -> VoiceStartConfiguration:
    provider_id = uuid4()
    return VoiceStartConfiguration(
        VoiceProviderConfiguration(
            provider_id,
            "qwen-audio",
            "Qwen Audio",
            VoiceProviderProtocol.QWEN_AUDIO_REALTIME_WS,
            "wss://workspace.example/api-ws/v1/realtime?region=cn",
            MappingProxyType({"api_key": "development-secret"}),
            MappingProxyType({"default_voice": "longanlingxi"}),
        ),
        VoiceModelConfiguration(
            uuid4(),
            provider_id,
            "audio",
            "qwen-audio-3.0-realtime-flash",
            "Qwen Audio 3.0",
            VoiceModelMode.NATIVE_REALTIME,
            MappingProxyType({"context_injection": True, "proactive_response": True}),
            MappingProxyType(audio_overrides),
            MappingProxyType({}),
            MappingProxyType({"connect_timeout_seconds": 4, "max_history_turns": 12}),
        ),
        VoiceChannelConfiguration(
            VoiceChannelKey("area", "voice"),
            "voice_readonly_v1",
            300,
        ),
        "",
        VoiceDuplexMode.FULL,
        None,
    )


def descriptor() -> VoiceSessionDescriptor:
    return VoiceSessionDescriptor(
        uuid4(),
        "person",
        VoiceChannelKey("area", "voice"),
        VoiceTextAddress("area", "text"),
    )


def test_qwen_audio_config_uses_documented_vad_and_nested_tool_schema() -> None:
    config = QwenAudioConfig.from_start_configuration(configuration())
    update = config.session_update(
        "short prompt",
        "event-1",
        (
            {
                "type": "function",
                "name": "delegate_agent_task",
                "description": "delegate",
                "parameters": {"type": "object"},
            },
        ),
    )

    assert config.turn_detection is QwenAudioTurnDetectionMode.SMART_TURN
    assert config.voice == "longanlingxi"
    assert config.max_history_turns == 12
    assert config.url.endswith("region=cn&model=qwen-audio-3.0-realtime-flash")
    session = update["session"]
    assert session["turn_detection"]["type"] == "smart_turn"
    assert session["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "delegate_agent_task",
                "description": "delegate",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert "development-secret" not in repr(config)
    assert "development-secret" not in json.dumps(update, ensure_ascii=False)


@pytest.mark.asyncio
async def test_configured_provider_builder_routes_qwen_audio_without_coordinator_branch() -> None:
    provider = ConfiguredVoiceProviderBuilder()(
        VoiceSessionRuntimeContext(
            descriptor(),
            SimpleNamespace(),
            configuration(),
        )
    )

    assert isinstance(provider, QwenAudioRealtimeProvider)
    await provider.aclose()


@pytest.mark.asyncio
async def test_qwen_audio_builder_compiles_memory_and_recovery_into_initial_instructions() -> None:
    provider = QwenAudioProviderBuilder()(
        VoiceSessionRuntimeContext(
            descriptor(),
            SimpleNamespace(),
            configuration(),
            memory_context="用户喜欢电子音乐。",
            recovery_context=VoiceRecoveryContext((VoiceRecoveryTurn("user", "刚才说到哪里了？"),)),
        )
    )

    assert "用户喜欢电子音乐" in provider._instructions
    assert '"role":"user","text":"刚才说到哪里了？"' in provider._instructions
    await provider.aclose()


@pytest.mark.asyncio
async def test_qwen_audio_adapter_injects_system_item_before_proactive_response() -> None:
    websocket = FakeWebSocket()
    provider = QwenAudioRealtimeProvider(
        QwenAudioConfig(
            "wss://workspace.example/realtime",
            "secret",
            "qwen-audio-3.0-realtime-flash",
        ),
        "prompt",
        connector=FakeConnector(websocket),
    )
    session = await provider.connect(descriptor())

    await session.request_proactive_response(
        VoiceInternalContextItem(
            "cywl_task_1",
            "[CYWL_INTERNAL_TASK_EVENT v1]\ntask: T1\nstate: succeeded",
        )
    )
    await session.finish()

    sent = [json.loads(payload) for payload in websocket.sent]
    assert [payload["type"] for payload in sent] == [
        "session.update",
        "conversation.item.create",
        "response.create",
    ]
    assert sent[1]["item"]["role"] == "system"
    assert sent[1]["item"]["content"][0]["type"] == "input_text"
    assert sent[1]["item"]["id"] == "cywl_task_1"
    assert provider.capabilities == VoiceProviderCapabilities(
        response_cancel=True,
        tool_calls=False,
        context_injection=True,
        proactive_response=True,
    )
    await provider.aclose()
    assert websocket.closed is True
