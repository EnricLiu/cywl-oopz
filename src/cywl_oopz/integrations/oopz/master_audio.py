"""OOPZ SDK adapter for the one canonical master PCM output."""

from __future__ import annotations

from oopz_sdk import OopzBot, PcmFormat, VoicePcmOutputStream, VoicePlaybackCursor

from cywl_oopz.features.audio.models import MASTER_AUDIO_FORMAT, MasterPlaybackCursor
from cywl_oopz.settings import AudioMixerSettings

MASTER_PREBUFFER_MS = 40
MASTER_MAX_BUFFER_MS = 160


class OopzMasterPcmOutput:
    """Map SDK output stats and ACK cursors to canonical master frames."""

    def __init__(self, stream: VoicePcmOutputStream) -> None:
        self._stream = stream

    @property
    def cursor(self) -> MasterPlaybackCursor:
        stats = self._stream.stats
        return MasterPlaybackCursor(
            stats.generation,
            stats.accepted_samples,
            stats.rendered_samples,
            stats.buffered_samples,
        )

    async def write(self, pcm_s16le: bytes) -> MasterPlaybackCursor:
        MASTER_AUDIO_FORMAT.frames_for_bytes(pcm_s16le)
        await self._stream.write(pcm_s16le)
        return self.cursor

    async def flush(self) -> MasterPlaybackCursor:
        return self._map_cursor(await self._stream.flush())

    async def drain(self) -> MasterPlaybackCursor:
        return self._map_cursor(await self._stream.drain())

    async def aclose(self) -> None:
        await self._stream.aclose()

    @staticmethod
    def _map_cursor(cursor: VoicePlaybackCursor) -> MasterPlaybackCursor:
        return MasterPlaybackCursor(
            cursor.generation,
            cursor.accepted_samples,
            cursor.rendered_samples,
            cursor.buffered_samples,
        )


class OopzMasterPcmOutputFactory:
    """Open the fixed 48 kHz stereo master contract on the SDK boundary."""

    def __init__(
        self,
        bot: OopzBot,
        *,
        prebuffer_ms: int = MASTER_PREBUFFER_MS,
        max_buffer_ms: int = MASTER_MAX_BUFFER_MS,
    ) -> None:
        if prebuffer_ms < 0 or max_buffer_ms <= 0 or prebuffer_ms > max_buffer_ms:
            raise ValueError("OOPZ master PCM buffer bounds are invalid")
        self._bot = bot
        self._prebuffer_ms = prebuffer_ms
        self._max_buffer_ms = max_buffer_ms

    @property
    def max_buffer_ms(self) -> int:
        return self._max_buffer_ms

    async def open(self) -> OopzMasterPcmOutput:
        stream = await self._bot.voice.open_pcm_output(
            PcmFormat.s16le(
                sample_rate=MASTER_AUDIO_FORMAT.sample_rate,
                channels=MASTER_AUDIO_FORMAT.channels,
            ),
            prebuffer_ms=self._prebuffer_ms,
            max_buffer_ms=self._max_buffer_ms,
        )
        return OopzMasterPcmOutput(stream)

    @classmethod
    def from_settings(
        cls,
        bot: OopzBot,
        settings: AudioMixerSettings,
    ) -> OopzMasterPcmOutputFactory:
        return cls(
            bot,
            prebuffer_ms=settings.master_prebuffer_ms,
            max_buffer_ms=settings.master_max_buffer_ms,
        )
