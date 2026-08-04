from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cywl_oopz.features.voice.audio import PROVIDER_INPUT_FORMAT
from cywl_oopz.features.voice.errors import VoiceProviderDisconnectedError
from cywl_oopz.features.voice.events import (
    VoiceProviderFailed,
    VoiceResponseCancelled,
    VoiceSessionFinished,
    VoiceSessionReady,
)
from cywl_oopz.features.voice.models import (
    PcmChunk,
    PlaybackCursor,
    VoiceChannelKey,
    VoiceRecoveryContext,
    VoiceRecoveryTurn,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.ports import VoiceSessionRuntimeContext
from cywl_oopz.integrations.voice.fake import FakeVoiceConfigurationRepository
from cywl_oopz.integrations.voice.qwen_omni import (
    QwenOmniProviderBuilder,
    QwenOmniRealtimeProvider,
)
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


class StalledCloseWebSocket(FakeWebSocket):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await self.allow_close.wait()
        await super().close()


class StalledConnector(FakeConnector):
    def __init__(self, websocket: FakeWebSocket) -> None:
        super().__init__(websocket)
        self.started = asyncio.Event()
        self.allow_connect = asyncio.Event()

    async def __call__(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        self.started.set()
        await self.allow_connect.wait()
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
        tool_schemas=(
            {
                "type": "function",
                "name": "delegate_agent_task",
                "parameters": {"type": "object"},
            },
        ),
        connector=connector,
    )

    session = await provider.connect(descriptor())
    await session.send_audio(PcmChunk(b"\x00" * 640, PROVIDER_INPUT_FORMAT, 20, 0))
    await session.complete_tool_call("call-1", {"ok": True, "task": "T1"})
    await session.finish()

    assert len(connector.calls) == 1
    url, options = connector.calls[0]
    assert url.endswith("?model=qwen3.5-omni-flash-realtime")
    assert options["additional_headers"]["Authorization"] == "Bearer development-secret"
    sent = [json.loads(payload) for payload in websocket.sent]
    assert [payload["type"] for payload in sent] == [
        "session.update",
        "input_audio_buffer.append",
        "conversation.item.create",
        "response.create",
        "session.finish",
    ]
    assert sent[0]["session"]["instructions"] == "short prompt"
    assert sent[0]["session"]["tools"][0]["name"] == "delegate_agent_task"
    assert provider.capabilities.tool_calls is True
    assert sent[2]["item"]["call_id"] == "call-1"
    assert json.loads(sent[2]["item"]["output"])["task"] == "T1"
    assert "development-secret" not in "".join(websocket.sent)
    assert len(sent[1]["audio"]) == 856

    await provider.aclose()
    assert websocket.closed is True


@pytest.mark.asyncio
async def test_qwen_omni_builder_compiles_recovery_into_initial_instructions() -> None:
    configuration = await FakeVoiceConfigurationRepository().resolve_start_configuration(
        "person", VoiceChannelKey("area", "voice")
    )
    provider = QwenOmniProviderBuilder()(
        VoiceSessionRuntimeContext(
            descriptor(),
            SimpleNamespace(),
            configuration,
            recovery_context=VoiceRecoveryContext(
                (VoiceRecoveryTurn("assistant", "上一句已经完整播完。"),)
            ),
        )
    )

    assert '"role":"assistant","text":"上一句已经完整播完。"' in provider._instructions
    await provider.aclose()


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


@pytest.mark.asyncio
async def test_qwen_adapter_cancels_only_an_active_response() -> None:
    websocket = FakeWebSocket()
    provider = QwenOmniRealtimeProvider(
        QwenOmniConfig("wss://workspace.example/realtime", "secret", "model"),
        "prompt",
        connector=FakeConnector(websocket),
    )
    session = await provider.connect(descriptor())
    events = session.events().__aiter__()
    await websocket.emit(
        {
            "type": "response.created",
            "response": {"id": "response-1", "status": "in_progress"},
        }
    )
    await anext(events)

    cursor = PlaybackCursor(1, 480, 240, 240, 24_000)
    await session.interrupt(cursor)
    assert [json.loads(payload)["type"] for payload in websocket.sent][-1] == "response.cancel"

    await websocket.emit(
        {
            "type": "response.done",
            "response": {"id": "response-1", "status": "cancelled"},
        }
    )
    assert isinstance(await anext(events), VoiceResponseCancelled)
    sent_count = len(websocket.sent)
    await session.interrupt(cursor)
    assert len(websocket.sent) == sent_count

    await provider.aclose()


@pytest.mark.asyncio
async def test_qwen_adapter_close_can_retry_after_caller_cancellation() -> None:
    websocket = StalledCloseWebSocket()
    provider = QwenOmniRealtimeProvider(
        QwenOmniConfig("wss://workspace.example/realtime", "secret", "model"),
        "prompt",
        connector=FakeConnector(websocket),
    )
    session = await provider.connect(descriptor())
    closing = asyncio.create_task(provider.aclose())
    await websocket.close_started.wait()

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    assert provider._closing is True
    assert provider._closed is False
    assert session._closing is True
    assert session._closed is False
    with pytest.raises(VoiceProviderDisconnectedError, match="closed"):
        await session.send_audio(PcmChunk(b"\x00" * 640, PROVIDER_INPUT_FORMAT, 20, 0))

    websocket.allow_close.set()
    await provider.aclose()

    assert websocket.closed is True
    assert websocket.close_calls == 2
    assert provider._closed is True
    assert session._closed is True


@pytest.mark.asyncio
async def test_qwen_adapter_rejects_connection_that_arrives_during_close() -> None:
    websocket = FakeWebSocket()
    connector = StalledConnector(websocket)
    provider = QwenOmniRealtimeProvider(
        QwenOmniConfig("wss://workspace.example/realtime", "secret", "model"),
        "prompt",
        connector=connector,
    )
    connecting = asyncio.create_task(provider.connect(descriptor()))
    await connector.started.wait()

    await provider.aclose()
    connector.allow_connect.set()

    with pytest.raises(VoiceProviderDisconnectedError, match="closed while connecting"):
        await connecting
    assert websocket.closed is True
    assert not provider._sessions
