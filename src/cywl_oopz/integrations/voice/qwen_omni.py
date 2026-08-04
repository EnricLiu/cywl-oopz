"""Async Qwen Omni Realtime WebSocket adapter behind project-owned voice ports."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Protocol
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.voice.audio import PROVIDER_INPUT_FORMAT
from cywl_oopz.features.voice.errors import (
    VoiceProviderAuthenticationError,
    VoiceProviderConfigurationError,
    VoiceProviderDisconnectedError,
    VoiceProviderError,
    VoiceProviderProtocolError,
    VoiceProviderRateLimitedError,
)
from cywl_oopz.features.voice.events import (
    VoiceModelEvent,
    VoiceProviderFailed,
    VoiceResponseCancelled,
    VoiceResponseCompleted,
    VoiceResponseStarted,
    VoiceSessionFinished,
)
from cywl_oopz.features.voice.models import (
    PcmChunk,
    PlaybackCursor,
    VoiceInternalContextItem,
    VoiceProviderCapabilities,
    VoiceSessionDescriptor,
)
from cywl_oopz.features.voice.ports import VoiceSessionRuntimeContext
from cywl_oopz.features.voice.prompt import VoicePromptCompiler

from .qwen_protocol import (
    QwenOmniConfig,
    audio_append_event,
    encode_client_event,
    function_call_output_event,
    internal_context_event,
    parse_server_event,
    response_cancel_event,
    response_create_event,
)

logger = logging.getLogger(__name__)

QwenConnector = Callable[..., Awaitable[ClientConnection]]


class QwenRealtimeWireConfig(Protocol):
    api_key: str
    model: str
    connect_timeout_seconds: float

    @property
    def url(self) -> str: ...

    def session_update(
        self,
        instructions: str,
        event_id: str,
        tools: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, object]: ...


class QwenOmniRealtimeSession:
    """One physical Qwen WebSocket with a serialized writer and single reader."""

    def __init__(
        self,
        websocket: ClientConnection,
        *,
        proactive_context: bool = False,
        send_finish_event: bool = True,
    ) -> None:
        self._websocket = websocket
        self._proactive_context = proactive_context
        self._send_finish_event = send_finish_event
        self._writer_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._events_started = False
        self._active_response_id = ""
        self._finishing = False
        self._closing = False
        self._closed = False

    async def send_audio(self, chunk: PcmChunk) -> None:
        if chunk.format != PROVIDER_INPUT_FORMAT:
            raise ValueError("Qwen input requires mono s16le 16 kHz PCM")
        await self._send(audio_append_event(chunk.pcm, _event_id()))

    async def configure(
        self,
        config: QwenRealtimeWireConfig,
        instructions: str,
        tools: Sequence[Mapping[str, object]],
    ) -> None:
        """Send the one initial session update before the event reader starts."""
        await self._send(config.session_update(instructions, _event_id(), tools))

    async def interrupt(self, cursor: PlaybackCursor) -> None:
        """Cancel generation; Qwen currently has no played-audio context truncate event."""
        del cursor
        if not self._active_response_id or self._closed or self._finishing:
            return
        await self._send(response_cancel_event(_event_id()))

    async def complete_tool_call(
        self,
        call_id: str,
        output: Mapping[str, object],
    ) -> None:
        """Return one function result and ask Qwen to continue the response."""
        await self._send_many(
            function_call_output_event(call_id, output, _event_id()),
            response_create_event(_event_id()),
        )

    async def request_proactive_response(self, item: VoiceInternalContextItem) -> None:
        if not self._proactive_context:
            raise VoiceProviderConfigurationError(
                "Qwen Provider does not support proactive context injection"
            )
        if self._active_response_id:
            raise VoiceProviderProtocolError("Qwen response is already in progress")
        await self._send_many(
            internal_context_event(item.item_id, item.text, _event_id()),
            response_create_event(_event_id()),
        )

    async def events(self) -> AsyncIterator[VoiceModelEvent]:
        if self._events_started:
            raise RuntimeError("Qwen event stream may only be consumed once")
        self._events_started = True
        finished = False
        try:
            async for raw in self._websocket:
                event = parse_server_event(raw)
                if event is not None:
                    if isinstance(event, VoiceResponseStarted):
                        self._active_response_id = event.response_id
                    elif isinstance(event, VoiceResponseCompleted | VoiceResponseCancelled):
                        if event.response_id == self._active_response_id:
                            self._active_response_id = ""
                    yield event
                if isinstance(event, VoiceSessionFinished):
                    finished = True
                    return
            if not finished and not self._finishing and not self._closing and not self._closed:
                yield VoiceProviderFailed("connection_eof", retryable=True)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as exc:
            if not self._finishing and not self._closing and not self._closed:
                yield VoiceProviderFailed(
                    error_kind="connection_closed",
                    retryable=exc.code not in {1000, 1001, 4001, 4003},
                )
        except VoiceProviderError as exc:
            yield VoiceProviderFailed(exception_kind(exc), retryable=False)
        except Exception as exc:
            logger.warning(
                "Qwen event reader failed: error=%s",
                exception_kind(exc),
            )
            yield VoiceProviderFailed(exception_kind(exc), retryable=True)

    async def finish(self) -> None:
        if self._closed or self._closing or self._finishing:
            return
        self._finishing = True
        if self._send_finish_event:
            await self._send({"event_id": _event_id(), "type": "session.finish"})

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            await self._websocket.close()
            self._closed = True

    async def _send(self, event: dict[str, object]) -> None:
        await self._send_many(event)

    async def _send_many(self, *events: dict[str, object]) -> None:
        if self._closed or self._closing:
            raise VoiceProviderDisconnectedError("Qwen session is closed")
        try:
            async with self._writer_lock:
                for event in events:
                    await self._websocket.send(encode_client_event(event))
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as exc:
            raise VoiceProviderDisconnectedError("Qwen WebSocket is closed") from exc
        except Exception as exc:
            raise VoiceProviderDisconnectedError("Qwen WebSocket write failed") from exc


class QwenOmniRealtimeProvider:
    """Create configured Qwen physical sessions without retaining conversation state."""

    def __init__(
        self,
        config: QwenRealtimeWireConfig,
        instructions: str,
        *,
        tool_schemas: Sequence[Mapping[str, object]] = (),
        connector: QwenConnector = connect,
        proactive_context: bool = False,
        send_finish_event: bool = True,
    ) -> None:
        self._config = config
        self._instructions = instructions
        self._tool_schemas = tuple(tool_schemas)
        self._connector = connector
        self._proactive_context = proactive_context
        self._send_finish_event = send_finish_event
        self._sessions: set[QwenOmniRealtimeSession] = set()
        self._close_lock = asyncio.Lock()
        self._closing = False
        self._closed = False

    @property
    def capabilities(self) -> VoiceProviderCapabilities:
        return VoiceProviderCapabilities(
            response_cancel=True,
            context_truncate_to_playout=False,
            tool_calls=bool(self._tool_schemas),
            context_injection=self._proactive_context,
            proactive_response=self._proactive_context,
        )

    async def connect(self, descriptor: VoiceSessionDescriptor) -> QwenOmniRealtimeSession:
        if self._closed or self._closing:
            raise VoiceProviderDisconnectedError("Qwen Provider is closed")
        logger.info(
            "Connecting Qwen realtime session: session=%s model=%s",
            opaque_ref(str(descriptor.session_id)),
            self._config.model,
        )
        try:
            async with asyncio.timeout(self._config.connect_timeout_seconds):
                websocket = await self._connector(
                    self._config.url,
                    additional_headers={"Authorization": f"Bearer {self._config.api_key}"},
                    open_timeout=self._config.connect_timeout_seconds,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=3,
                    max_size=2 * 1024 * 1024,
                )
        except asyncio.CancelledError:
            raise
        except InvalidStatus as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                raise VoiceProviderAuthenticationError("Qwen authentication failed") from exc
            if status == 429:
                raise VoiceProviderRateLimitedError("Qwen rate limit exceeded") from exc
            raise VoiceProviderDisconnectedError("Qwen handshake was rejected") from exc
        except TimeoutError as exc:
            raise VoiceProviderDisconnectedError("Qwen connection timed out") from exc
        except Exception as exc:
            raise VoiceProviderDisconnectedError("Qwen connection failed") from exc

        session = QwenOmniRealtimeSession(
            websocket,
            proactive_context=self._proactive_context,
            send_finish_event=self._send_finish_event,
        )
        try:
            await session.configure(self._config, self._instructions, self._tool_schemas)
        except BaseException:
            await session.aclose()
            raise
        async with self._close_lock:
            if not self._closing and not self._closed:
                self._sessions.add(session)
                return session
        await session.aclose()
        raise VoiceProviderDisconnectedError("Qwen Provider closed while connecting")

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            sessions = tuple(self._sessions)
            await asyncio.gather(*(session.aclose() for session in sessions))
            self._sessions.clear()
            self._closed = True


class QwenOmniProviderBuilder:
    """Parse one pinned DB configuration and build its stateless Qwen adapter."""

    def __init__(
        self,
        prompt_compiler: VoicePromptCompiler | None = None,
        *,
        tool_schemas: Sequence[Mapping[str, object]] = (),
    ) -> None:
        self._prompts = prompt_compiler or VoicePromptCompiler()
        self._tool_schemas = tuple(tool_schemas)

    def __call__(self, context: VoiceSessionRuntimeContext) -> QwenOmniRealtimeProvider:
        configuration = context.configuration
        return QwenOmniRealtimeProvider(
            QwenOmniConfig.from_start_configuration(configuration),
            self._prompts.compile(
                configuration,
                memory_context=context.memory_context,
                recovery_context=context.recovery_context,
            ),
            tool_schemas=self._tool_schemas,
        )


def _event_id() -> str:
    return f"event_{uuid4().hex}"
