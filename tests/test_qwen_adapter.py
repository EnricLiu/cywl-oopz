from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest

from cywl_oopz.features.voice.audio import PROVIDER_INPUT_FORMAT
from cywl_oopz.features.voice.events import (
    VoiceProviderFailed,
    VoiceSessionFinished,
    VoiceSessionReady,
)
from cywl_oopz.features.voice.models import (
    PcmChunk,
    VoiceChannelKey,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.integrations.voice.qwen_omni import QwenOmniRealtimeProvider
from cywl_oopz.integrations.voice.qwen_protocol import QwenOmniConfig


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

    async def emit(self, payload: dict) -> None:
        await self.incoming.put(json.dumps(payload))


class FakeConnector:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return self.websocket


def descriptor() -> VoiceSessionDescriptor:
    return VoiceSessionDescriptor(
        uuid4(),
        "person",
        VoiceChannelKey("area", "voice"),
        VoiceTextAddress("area", "text"),
    )


@pytest.mark.asyncio
async def test_qwen_adapter_serializes_session_update_audio_and_finish() -> None:
    websocket = FakeWebSocket()
    connector = FakeConnector(websocket)
    config = QwenOmniConfig(
        "wss://workspace.example/realtime",
        "development-secret",
        "qwen3.5-omni-flash-realtime",
    )
    provider = QwenOmniRealtimeProvider(
        config,
        "short prompt",
        connector=connector,
    )

    session = await provider.connect(descriptor())
    await session.send_audio(PcmChunk(b"\x00" * 640, PROVIDER_INPUT_FORMAT, 20, 0))
    await session.finish()

    assert len(connector.calls) == 1
    url, options = connector.calls[0]
    assert url.endswith("?model=qwen3.5-omni-flash-realtime")
    assert options["additional_headers"]["Authorization"] == "Bearer development-secret"
    sent = [json.loads(payload) for payload in websocket.sent]
    assert [payload["type"] for payload in sent] == [
        "session.update",
        "input_audio_buffer.append",
        "session.finish",
    ]
    assert sent[0]["session"]["instructions"] == "short prompt"
    assert "development-secret" not in "".join(websocket.sent)
    assert len(sent[1]["audio"]) == 856

    await provider.aclose()
    assert websocket.closed is True


@pytest.mark.asyncio
async def test_qwen_adapter_yields_ready_and_clean_terminal_once() -> None:
    websocket = FakeWebSocket()
    provider = QwenOmniRealtimeProvider(
        QwenOmniConfig("wss://workspace.example/realtime", "secret", "model"),
        "prompt",
        connector=FakeConnector(websocket),
    )
    session = await provider.connect(descriptor())
    await websocket.emit({"type": "session.updated"})
    await websocket.emit({"type": "session.finished"})

    events = [event async for event in session.events()]

    assert isinstance(events[0], VoiceSessionReady)
    assert isinstance(events[1], VoiceSessionFinished)
    await provider.aclose()


@pytest.mark.asyncio
async def test_qwen_adapter_sanitizes_malformed_wire_event() -> None:
    websocket = FakeWebSocket()
    provider = QwenOmniRealtimeProvider(
        QwenOmniConfig("wss://workspace.example/realtime", "secret", "model"),
        "prompt",
        connector=FakeConnector(websocket),
    )
    session = await provider.connect(descriptor())
    await websocket.incoming.put("not-json")

    events = [event async for event in session.events()]

    assert events == [VoiceProviderFailed("VoiceProviderProtocolError", False)]
    await provider.aclose()
