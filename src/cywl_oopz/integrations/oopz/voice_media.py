"""Map OOPZ realtime voice media handles to project-owned ports and values."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from oopz_sdk import (
    OopzBot,
    PcmFormat,
    VoiceAudioEndReason,
    VoiceAudioSubscription,
    VoicePcmOutputStream,
    VoicePlaybackCursor,
)

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.voice.audio import PROVIDER_OUTPUT_FORMAT
from cywl_oopz.features.voice.errors import VoiceMediaTransportError
from cywl_oopz.features.voice.models import (
    PcmChunk,
    PlaybackCursor,
    RemoteAudioFrame,
    VoiceAudioFormat,
    VoiceMediaEndReason,
    VoiceMediaTerminal,
    VoiceSessionDescriptor,
)
from cywl_oopz.features.voice.ports import VoiceLease
from cywl_oopz.settings import VoiceSettings

logger = logging.getLogger(__name__)

_INPUT_FRAME_SIZE = 1024
_INPUT_QUEUE_SIZE = 8

_END_REASON_MAP = {
    VoiceAudioEndReason.CLOSED_BY_CALLER: VoiceMediaEndReason.CLOSED_BY_CALLER,
    VoiceAudioEndReason.REMOTE_UNPUBLISHED: VoiceMediaEndReason.OWNER_UNPUBLISHED,
    VoiceAudioEndReason.REMOTE_LEFT: VoiceMediaEndReason.OWNER_LEFT,
    VoiceAudioEndReason.VOICE_LEFT: VoiceMediaEndReason.VOICE_LEFT,
    VoiceAudioEndReason.BACKEND_CLOSED: VoiceMediaEndReason.BACKEND_CLOSED,
    VoiceAudioEndReason.TRANSPORT_LOST: VoiceMediaEndReason.TRANSPORT_LOST,
    VoiceAudioEndReason.QUEUE_OVERFLOW: VoiceMediaEndReason.QUEUE_OVERFLOW,
}


class OopzVoiceMediaSession:
    """Own one person subscription and one incremental PCM output handle."""

    def __init__(
        self,
        subscription: VoiceAudioSubscription,
        output: VoicePcmOutputStream,
    ) -> None:
        self._subscription = subscription
        self._output = output
        self._close_lock = asyncio.Lock()
        self._closing = False
        self._subscription_closed = False
        self._output_closed = False
        self._closed = False

    async def input_frames(self) -> AsyncIterator[RemoteAudioFrame]:
        try:
            async for frame in self._subscription:
                yield RemoteAudioFrame(
                    pcm=frame.data,
                    format=VoiceAudioFormat(
                        frame.sample_rate,
                        frame.channels,
                        frame.sample_format,
                    ),
                    sequence=frame.sequence,
                    captured_at_monotonic=frame.received_at_monotonic,
                    source_dropped_frames=frame.browser_dropped_before,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._transport_error("input_frames", exc) from exc

    async def wait_input_closed(self) -> VoiceMediaTerminal:
        try:
            reason = await self._subscription.wait_closed()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._transport_error("wait_input_closed", exc) from exc
        return VoiceMediaTerminal(
            _END_REASON_MAP[reason],
            (
                exception_kind(self._subscription.terminal_error)
                if self._subscription.terminal_error is not None
                else None
            ),
        )

    async def write_output(self, chunk: PcmChunk) -> PlaybackCursor:
        if chunk.format != PROVIDER_OUTPUT_FORMAT:
            raise ValueError("OOPZ voice output requires mono s16le 24 kHz PCM")
        try:
            await self._output.write(chunk.pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._transport_error("write_output", exc) from exc
        return self._cursor_from_stats()

    async def flush_output(self) -> PlaybackCursor:
        try:
            return self._map_cursor(await self._output.flush())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._transport_error("flush_output", exc) from exc

    async def drain_output(self) -> PlaybackCursor:
        try:
            return self._map_cursor(await self._output.drain())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._transport_error("drain_output", exc) from exc

    async def current_cursor(self) -> PlaybackCursor:
        return self._cursor_from_stats()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            operations = []
            names = []
            if not self._subscription_closed:
                operations.append(self._close_subscription())
                names.append("owner audio subscription")
            if not self._output_closed:
                operations.append(self._close_output())
                names.append("PCM output")
            results = await asyncio.gather(*operations, return_exceptions=True)
            for name, result in zip(names, results, strict=True):
                if not isinstance(result, BaseException):
                    continue
                logger.warning(
                    "Could not close OOPZ %s: error=%s",
                    name,
                    exception_kind(result),
                )
            self._closed = self._subscription_closed and self._output_closed

    async def _close_subscription(self) -> None:
        await self._subscription.aclose()
        self._subscription_closed = True

    async def _close_output(self) -> None:
        await self._output.aclose()
        self._output_closed = True

    def _cursor_from_stats(self) -> PlaybackCursor:
        stats = self._output.stats
        return PlaybackCursor(
            generation=stats.generation,
            accepted_samples=stats.accepted_samples,
            rendered_samples=stats.rendered_samples,
            buffered_samples=stats.buffered_samples,
            sample_rate=self._output.format.sample_rate,
        )

    @staticmethod
    def _map_cursor(cursor: VoicePlaybackCursor) -> PlaybackCursor:
        return PlaybackCursor(
            generation=cursor.generation,
            accepted_samples=cursor.accepted_samples,
            rendered_samples=cursor.rendered_samples,
            buffered_samples=cursor.buffered_samples,
            sample_rate=cursor.sample_rate,
        )

    @staticmethod
    def _transport_error(operation: str, error: Exception) -> VoiceMediaTransportError:
        return VoiceMediaTransportError(operation, exception_kind(error))


class OopzVoiceMediaGateway:
    """Open only the session owner's audio and the bounded bot PCM output."""

    def __init__(self, bot: OopzBot, settings: VoiceSettings) -> None:
        self._bot = bot
        self._settings = settings

    async def open(
        self,
        descriptor: VoiceSessionDescriptor,
        lease: VoiceLease,
    ) -> OopzVoiceMediaSession:
        if lease.released:
            raise ValueError("Voice media requires an active voice lease")
        logger.info(
            "Opening OOPZ conversation media: session=%s owner=%s channel=%s",
            opaque_ref(str(descriptor.session_id)),
            opaque_ref(descriptor.owner_person_id),
            opaque_ref(
                descriptor.voice_channel.area_id,
                descriptor.voice_channel.channel_id,
            ),
        )
        try:
            subscription = await self._bot.voice.subscribe_person_audio(
                descriptor.owner_person_id,
                frame_size=_INPUT_FRAME_SIZE,
                max_queue_size=_INPUT_QUEUE_SIZE,
                wait_timeout=self._settings.start_timeout_seconds,
                force_profile=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise VoiceMediaTransportError("open_input", exception_kind(exc)) from exc

        try:
            output = await self._bot.voice.open_pcm_output(
                PcmFormat.s16le(
                    sample_rate=PROVIDER_OUTPUT_FORMAT.sample_rate,
                    channels=PROVIDER_OUTPUT_FORMAT.channels,
                ),
                prebuffer_ms=self._settings.output_prebuffer_ms,
                max_buffer_ms=self._settings.output_queue_ms,
            )
        except asyncio.CancelledError:
            await asyncio.shield(self._close_failed_subscription(subscription))
            raise
        except Exception as exc:
            await self._close_failed_subscription(subscription)
            raise VoiceMediaTransportError("open_output", exception_kind(exc)) from exc
        return OopzVoiceMediaSession(subscription, output)

    @staticmethod
    async def _close_failed_subscription(subscription: VoiceAudioSubscription) -> None:
        try:
            await subscription.aclose()
        except Exception as exc:
            logger.warning(
                "Could not close OOPZ input after output open failure: error=%s",
                exception_kind(exc),
            )
