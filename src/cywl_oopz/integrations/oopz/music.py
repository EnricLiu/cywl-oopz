"""OOPZ SDK adapter for project-owned music playback use cases."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from oopz_sdk import OopzBot, VoicePlayback

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.audio.models import (
    AudioChannelKey,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)
from cywl_oopz.features.audio.ports import AudioDecoder
from cywl_oopz.features.audio.session import SharedAudioMixerBus
from cywl_oopz.features.music.models import (
    MusicPlaybackEndReason,
    MusicPlaybackResult,
    VoiceChannelKey,
)
from cywl_oopz.integrations.audio.ffmpeg import FfmpegMusicDecoderFactory
from cywl_oopz.integrations.audio.music import FfmpegMusicPlayback
from cywl_oopz.settings import AudioMixerSettings

from .voice_channel_session import (
    OopzVoiceChannelSessionManager,
    OopzVoiceParticipant,
)

DEFAULT_VOLUME = 2  # %

logger = logging.getLogger(__name__)


class _MusicDecoderFactory(Protocol):
    async def validate(self) -> None: ...

    async def open(self, stream_url: str) -> AudioDecoder: ...


class OopzMusicPlayback:
    """Map one SDK playback handle to the project music port."""

    def __init__(self, playback: VoicePlayback) -> None:
        self._playback = playback

    @property
    def finished(self) -> bool:
        return self._playback.finished

    async def wait_finished(self) -> MusicPlaybackResult:
        result = await self._playback.wait_finished()
        return MusicPlaybackResult(
            end_reason=MusicPlaybackEndReason(result.end_reason.value),
            duration_seconds=result.duration_seconds,
            terminal_error=result.terminal_error,
        )

    async def stop(self) -> None:
        await self._playback.stop()

    async def pause(self) -> bool:
        await self._playback.pause()
        return True

    async def resume(self) -> bool:
        await self._playback.resume()
        return True


class OopzMusicVoiceGateway:
    """Translate music operations under one shared OOPZ voice lease."""

    def __init__(
        self,
        bot: OopzBot,
        leases: OopzVoiceChannelSessionManager,
        audio_settings: AudioMixerSettings | None = None,
        *,
        decoder_factory: _MusicDecoderFactory | None = None,
    ) -> None:
        self._bot = bot
        self._leases = leases
        self._audio_settings = audio_settings or AudioMixerSettings.from_mapping({})
        self._decoder_factory = decoder_factory or FfmpegMusicDecoderFactory(self._audio_settings)
        self._lease: OopzVoiceParticipant | None = None
        self._channel: VoiceChannelKey | None = None
        self._playback: OopzMusicPlayback | FfmpegMusicPlayback | None = None
        self._bus: SharedAudioMixerBus | None = None
        self._lock = asyncio.Lock()

    async def validate_capabilities(self) -> None:
        if self._audio_settings.enabled:
            await self._decoder_factory.validate()

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        channel = await self._bot.channels.get_voice_channel_for_user(area_id, person_id)
        logger.debug(
            "Resolved user voice channel: area=%s user=%s found=%s",
            opaque_ref(area_id),
            opaque_ref(person_id),
            channel is not None,
        )
        return channel

    async def acquire(self, channel: VoiceChannelKey) -> bool:
        async with self._lock:
            if self._lease is not None:
                return self._channel == channel and not self._lease.released
            lease = await self._leases.try_acquire(
                VoiceParticipantRequest(
                    VoiceParticipantKind.MUSIC,
                    AudioChannelKey(channel.area_id, channel.channel_id),
                    owner_key=f"music:{channel.area_id}:{channel.channel_id}",
                )
            )
            if lease is None:
                return False
            self._lease = lease
            self._channel = channel
            return True

    async def start_playback(
        self,
        channel: VoiceChannelKey,
        stream_url: str,
    ) -> OopzMusicPlayback | FfmpegMusicPlayback:
        async with self._lock:
            if self._lease is None or self._lease.released or self._channel != channel:
                raise RuntimeError("Music playback requires a matching active voice lease")
            if self._playback is not None and not self._playback.finished:
                raise RuntimeError("Music playback already has an active owner handle")
            logger.debug(
                "Starting typed OOPZ music playback: channel=%s",
                self._channel_ref(channel),
            )
            if self._audio_settings.enabled:
                bus = await self._ensure_bus_locked()
                decoder = await self._decoder_factory.open(stream_url)
                playback = FfmpegMusicPlayback.from_bus(decoder, bus)
                self._playback = playback
                return playback
            await self._bot.voice.set_volume(DEFAULT_VOLUME)
            playback = OopzMusicPlayback(await self._bot.voice.start_url_playback(stream_url))
            self._playback = playback
            return playback

    async def release(self, channel: VoiceChannelKey) -> bool:
        async with self._lock:
            if self._lease is None or self._channel != channel:
                return False
            playback = self._playback
            lease = self._lease
            if playback is not None and not playback.finished:
                try:
                    await playback.stop()
                except Exception as exc:
                    logger.warning(
                        "Could not stop music before releasing voice lease: error=%s",
                        exception_kind(exc),
                    )
                else:
                    self._playback = None
            released = await lease.release()
            if lease.released:
                self._clear_lease_locked()
            return released

    async def aclose(self) -> None:
        async with self._lock:
            playback = self._playback
            lease = self._lease
            if playback is not None and not playback.finished:
                try:
                    await playback.stop()
                except Exception as exc:
                    logger.warning(
                        "Could not stop music during close: error=%s",
                        exception_kind(exc),
                    )
                else:
                    self._playback = None
            if lease is not None:
                try:
                    await lease.release()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Could not release music voice lease during close: error=%s",
                        exception_kind(exc),
                    )
                    return
                if lease.released:
                    self._clear_lease_locked()
            else:
                self._channel = None
        logger.info("Closed OOPZ music gateway")

    def _clear_lease_locked(self) -> None:
        self._playback = None
        self._bus = None
        self._lease = None
        self._channel = None

    async def _ensure_bus_locked(self) -> SharedAudioMixerBus:
        if self._bus is not None and not self._bus.closed:
            return self._bus
        if self._lease is None or self._lease.released:
            raise RuntimeError("Music PCM output requires an active voice participant")
        self._bus = await self._lease.audio_bus()
        return self._bus

    @staticmethod
    def _channel_ref(channel: VoiceChannelKey) -> str:
        return opaque_ref(channel.area_id, channel.channel_id)
