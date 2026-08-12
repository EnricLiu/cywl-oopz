"""OOPZ SDK adapter for project-owned music playback use cases."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Protocol

from oopz_sdk import OopzBot, VoicePlayback

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.audio.errors import AudioBusFailedError
from cywl_oopz.features.audio.models import (
    AudioChannelKey,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)
from cywl_oopz.features.audio.ports import AudioDecoder
from cywl_oopz.features.audio.session import SharedAudioMixerBus
from cywl_oopz.features.music.errors import MusicBackendClosedError, MusicPlaybackError
from cywl_oopz.features.music.models import (
    MusicPlaybackEndReason,
    MusicPlaybackResult,
    PlayableTrack,
    ResolvedMediaInput,
    VoiceChannelKey,
)
from cywl_oopz.integrations.audio.ffmpeg import FfmpegMusicDecoderFactory
from cywl_oopz.integrations.audio.music import FfmpegMusicPlayback
from cywl_oopz.settings import AudioMixerSettings

from .voice_channel_session import (
    OopzVoiceChannelSessionManager,
    OopzVoiceParticipant,
    VoiceParticipantTerminationReason,
)

DEFAULT_VOLUME = 2  # %

logger = logging.getLogger(__name__)


class _MusicDecoderFactory(Protocol):
    async def validate(self) -> None: ...

    async def open(self, media: ResolvedMediaInput) -> AudioDecoder: ...


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


class ObservedMusicPlayback:
    """Complete playback when either media or its physical voice generation ends."""

    def __init__(
        self,
        playback: OopzMusicPlayback | FfmpegMusicPlayback,
        participant: OopzVoiceParticipant,
    ) -> None:
        self._playback = playback
        self._participant = participant
        self._result = asyncio.create_task(
            self._observe(),
            name=f"music-playback-generation:{participant.generation}",
        )

    @property
    def finished(self) -> bool:
        return self._result.done()

    async def wait_finished(self) -> MusicPlaybackResult:
        return await asyncio.shield(self._result)

    async def stop(self) -> None:
        if self.finished:
            return
        await self._playback.stop()

    async def pause(self) -> bool:
        if self.finished:
            return False
        return await self._playback.pause()

    async def resume(self) -> bool:
        if self.finished:
            return False
        return await self._playback.resume()

    async def _observe(self) -> MusicPlaybackResult:
        playback_wait = asyncio.create_task(
            self._playback.wait_finished(),
            name="music-playback-terminal",
        )
        participant_wait = asyncio.create_task(
            self._participant.wait_terminated(),
            name=f"music-participant-terminal:{self._participant.generation}",
        )
        try:
            done, _pending = await asyncio.wait(
                {playback_wait, participant_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if playback_wait in done:
                return playback_wait.result()
            termination = participant_wait.result()
            with suppress(Exception):
                async with asyncio.timeout(1.0):
                    await self._playback.stop()
            reason = (
                MusicPlaybackEndReason.VOICE_LEFT
                if termination.reason
                in {
                    VoiceParticipantTerminationReason.BACKEND_FAILED,
                    VoiceParticipantTerminationReason.MANAGER_CLOSED,
                }
                else MusicPlaybackEndReason.STOPPED
            )
            return MusicPlaybackResult(reason)
        finally:
            for waiting in (playback_wait, participant_wait):
                if not waiting.done():
                    waiting.cancel()
            await asyncio.gather(playback_wait, participant_wait, return_exceptions=True)


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
        self._playback: ObservedMusicPlayback | None = None
        self._bus: SharedAudioMixerBus | None = None
        self._backend_recovery_pending = False
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
                if self._lease.released:
                    self._clear_lease_locked()
                else:
                    return self._channel == channel
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
        playable: PlayableTrack,
    ) -> ObservedMusicPlayback:
        async with self._lock:
            if self._lease is None or self._lease.released or self._channel != channel:
                raise RuntimeError("Music playback requires a matching active voice lease")
            if self._playback is not None and not self._playback.finished:
                raise RuntimeError("Music playback already has an active owner handle")
            logger.debug(
                "Starting typed OOPZ music playback: channel=%s source=%s",
                self._channel_ref(channel),
                playable.track.source.value,
            )
            if self._audio_settings.enabled:
                restarting = self._backend_recovery_pending or (
                    self._bus is not None and self._bus.failed
                )
                decoder: AudioDecoder | None = None
                try:
                    bus = await self._ensure_bus_locked()
                    started_at = time.monotonic()
                    decoder = await self._decoder_factory.open(playable.media)
                    elapsed_ms = (time.monotonic() - started_at) * 1_000
                    await bus.record_decoder_start(elapsed_ms, restarted=restarting)
                    self._backend_recovery_pending = False
                except BaseException as exc:
                    if decoder is not None:
                        with suppress(BaseException):
                            await asyncio.shield(decoder.aclose())
                    if isinstance(exc, AudioBusFailedError):
                        self._backend_recovery_pending = True
                        raise MusicBackendClosedError(
                            "Shared music audio backend closed during startup"
                        ) from exc
                    raise
                source_playback = FfmpegMusicPlayback.from_bus(decoder, bus)
                playback = ObservedMusicPlayback(source_playback, self._lease)
                self._playback = playback
                return playback
            if playable.media.http_headers:
                raise MusicPlaybackError("Music media HTTP headers require the shared audio mixer")
            await self._bot.voice.set_volume(DEFAULT_VOLUME)
            source_playback = OopzMusicPlayback(
                await self._bot.voice.start_url_playback(playable.media.url)
            )
            playback = ObservedMusicPlayback(source_playback, self._lease)
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

    async def reset(self, channel: VoiceChannelKey) -> None:
        """Invalidate only the matching stale physical generation."""
        async with self._lock:
            if self._lease is None or self._channel != channel:
                return
            lease = self._lease
            playback = self._playback
        if playback is not None and not playback.finished:
            with suppress(Exception):
                await playback.stop()
        if not lease.released:
            await self._leases.invalidate_backend(expected_generation=lease.generation)
        async with self._lock:
            if self._lease is lease and lease.released:
                self._clear_lease_locked()

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
        self._backend_recovery_pending = False
        self._lease = None
        self._channel = None

    async def _ensure_bus_locked(self) -> SharedAudioMixerBus:
        if self._bus is not None and not self._bus.closed and not self._bus.failed:
            return self._bus
        if self._lease is None or self._lease.released:
            raise RuntimeError("Music PCM output requires an active voice participant")
        self._bus = await self._lease.audio_bus()
        return self._bus

    @staticmethod
    def _channel_ref(channel: VoiceChannelKey) -> str:
        return opaque_ref(channel.area_id, channel.channel_id)
