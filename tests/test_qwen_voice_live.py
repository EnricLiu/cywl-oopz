from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv
from oopz_sdk import OopzBot, OopzConfig
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.audio.models import (
    AudioChannelKey,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)
from cywl_oopz.features.music.errors import MusicNotFoundError
from cywl_oopz.features.music.models import (
    MusicPlaybackEndReason,
    PlayableTrack,
)
from cywl_oopz.features.music.models import (
    VoiceChannelKey as MusicVoiceChannelKey,
)
from cywl_oopz.features.music.netease import NeteaseMusicCatalog
from cywl_oopz.features.music.ports import MusicPlayback
from cywl_oopz.features.voice.errors import VoiceProviderConfigurationError
from cywl_oopz.features.voice.events import VoiceSessionFinished, VoiceSessionReady
from cywl_oopz.features.voice.models import (
    RemoteAudioFrame,
    VoiceAudioFormat,
    VoiceChannelKey,
    VoiceMediaEndReason,
    VoiceSessionDescriptor,
    VoiceSessionState,
    VoiceStopReason,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.ports import VoiceLease, VoiceSessionRuntimeContext
from cywl_oopz.features.voice.runtime import RealtimeVoiceSessionRuntimeImpl
from cywl_oopz.features.voice.settings import (
    VoiceChannelConfiguration,
    VoiceDuplexMode,
    VoiceModelConfiguration,
    VoiceProviderConfiguration,
    VoiceProviderProtocol,
    VoiceStartConfiguration,
    VoiceTurnRole,
)
from cywl_oopz.integrations.oopz.master_audio import OopzMasterPcmOutputFactory
from cywl_oopz.integrations.oopz.music import OopzMusicVoiceGateway
from cywl_oopz.integrations.oopz.voice_channel_session import (
    OopzVoiceChannelSessionManager,
    OopzVoiceParticipant,
)
from cywl_oopz.integrations.oopz.voice_media import OopzVoiceMediaGateway
from cywl_oopz.integrations.voice.fake import (
    FakeVoiceAccessGateway,
    FakeVoiceConfigurationRepository,
    FakeVoiceMediaGateway,
    FakeVoiceMediaSession,
    FakeVoiceSessionRepository,
)
from cywl_oopz.integrations.voice.qwen_omni import QwenOmniRealtimeProvider
from cywl_oopz.integrations.voice.qwen_protocol import QwenOmniConfig
from cywl_oopz.settings import AudioMixerSettings, MusicSettings, VoiceSettings
from cywl_oopz.storage.models import VoiceModelRecord, VoiceProviderRecord
from cywl_oopz.storage.url import normalize_asyncpg_url

_LIVE_FRAME_SAMPLES = 1_024
logger = logging.getLogger(__name__)


def _bounded_live_integer(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _bounded_live_soak_seconds() -> int:
    name = "CYWL_VOICE_MUSIC_SOAK_SECONDS"
    raw = os.getenv(name, "1800").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 60 <= value <= 1_800:
        raise ValueError(f"{name} must be between 60 and 1800")
    return value


@dataclass(frozen=True, slots=True)
class _LiveWavReplay:
    """A bounded local speech fixture converted to OOPZ-equivalent float frames."""

    format: VoiceAudioFormat
    chunks: tuple[bytes, ...]

    @classmethod
    def load(cls, path: Path) -> _LiveWavReplay:
        if not path.is_file():
            pytest.fail("CYWL_QWEN_LIVE_INPUT_WAV does not point to a file")
        try:
            with wave.open(str(path), "rb") as source:
                channels = source.getnchannels()
                sample_rate = source.getframerate()
                sample_width = source.getsampwidth()
                frame_count = source.getnframes()
                compression = source.getcomptype()
                pcm = source.readframes(frame_count)
        except (OSError, wave.Error) as exc:
            pytest.fail(f"could not read CYWL_QWEN_LIVE_INPUT_WAV: {type(exc).__name__}")
        duration_seconds = frame_count / sample_rate if sample_rate else 0
        if compression != "NONE" or sample_width != 2 or channels not in {1, 2}:
            pytest.fail("Qwen live input WAV must be uncompressed 16-bit mono or stereo PCM")
        if sample_rate not in {16_000, 24_000, 44_100, 48_000}:
            pytest.fail("Qwen live input WAV sample rate must be 16, 24, 44.1, or 48 kHz")
        if not 0.2 <= duration_seconds <= 15:
            pytest.fail("Qwen live input WAV duration must be between 0.2 and 15 seconds")
        samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels)
        floats = (samples.astype(np.float32) / 32768.0).astype("<f4", copy=False)
        chunks = tuple(
            floats[index : index + _LIVE_FRAME_SAMPLES].tobytes()
            for index in range(0, len(floats), _LIVE_FRAME_SAMPLES)
        )
        return cls(VoiceAudioFormat(sample_rate, channels, "f32le"), chunks)

    async def replay(self, media, sequence: int, *, trailing_silence_ms: int) -> int:
        frame_width = self.format.frame_width_bytes
        for pcm in self.chunks:
            await media.push_input(RemoteAudioFrame(pcm, self.format, sequence, time.monotonic()))
            sequence += 1
            await asyncio.sleep((len(pcm) // frame_width) / self.format.sample_rate)
        remaining = self.format.sample_rate * trailing_silence_ms // 1_000
        while remaining:
            samples = min(_LIVE_FRAME_SAMPLES, remaining)
            pcm = bytes(samples * frame_width)
            await media.push_input(RemoteAudioFrame(pcm, self.format, sequence, time.monotonic()))
            sequence += 1
            remaining -= samples
            await asyncio.sleep(samples / self.format.sample_rate)
        return sequence


@dataclass(slots=True)
class _QwenLiveRuntimeHarness:
    """Own one paid Provider runtime and all deterministic local media resources."""

    runtime: RealtimeVoiceSessionRuntimeImpl
    media: FakeVoiceMediaSession
    sessions: FakeVoiceSessionRepository
    providers: list[QwenOmniRealtimeProvider]
    lease: VoiceLease

    @classmethod
    async def create(
        cls,
        config: QwenOmniConfig,
        instructions: str,
        *,
        idle_timeout_seconds: int = 300,
        owner_leave_grace_seconds: int = 15,
    ) -> _QwenLiveRuntimeHarness:
        access = FakeVoiceAccessGateway()
        channel = VoiceChannelKey("live-test-area", "live-test-voice")
        lease = await access.try_acquire(channel, "qwen-live-runtime")
        assert lease is not None
        base_configuration = await FakeVoiceConfigurationRepository().resolve_start_configuration(
            "live-test-person", channel
        )
        configuration = replace(
            base_configuration,
            channel=replace(
                base_configuration.channel,
                idle_timeout_seconds=idle_timeout_seconds,
            ),
        )
        descriptor = VoiceSessionDescriptor(
            uuid4(),
            "live-test-person",
            channel,
            VoiceTextAddress("live-test-area", "live-test-text"),
        )
        media_gateway = FakeVoiceMediaGateway()
        sessions = FakeVoiceSessionRepository()
        providers: list[QwenOmniRealtimeProvider] = []

        def build_provider(context):
            del context
            provider = QwenOmniRealtimeProvider(config, instructions)
            providers.append(provider)
            return provider

        runtime = RealtimeVoiceSessionRuntimeImpl(
            VoiceSessionRuntimeContext(descriptor, lease, configuration),
            VoiceSettings.from_mapping(
                {
                    "CYWL_VOICE_ENABLED": "true",
                    "CYWL_VOICE_START_TIMEOUT_SECONDS": "30",
                    "CYWL_VOICE_STOP_TIMEOUT_SECONDS": "1.5",
                    "CYWL_VOICE_IDLE_TIMEOUT_SECONDS": "300",
                    "CYWL_VOICE_OWNER_LEAVE_GRACE_SECONDS": str(owner_leave_grace_seconds),
                    "CYWL_VOICE_MAX_SESSION_SECONDS": "1800",
                    "CYWL_VOICE_INPUT_QUEUE_MS": "1000",
                    "CYWL_VOICE_OUTPUT_QUEUE_MS": "2000",
                }
            ),
            media_gateway,
            sessions,
            build_provider,
        )
        try:
            await runtime.start()
        except BaseException:
            await runtime.aclose()
            await lease.release()
            raise
        return cls(runtime, media_gateway.sessions[0], sessions, providers, lease)

    async def stop(self) -> None:
        await self.runtime.request_stop(VoiceStopReason.COMMAND)
        async with asyncio.timeout(2):
            result = await self.runtime.wait_finished()
        assert result.reason is VoiceStopReason.COMMAND

    async def aclose(self) -> None:
        await self.runtime.aclose()
        await self.lease.release()


@dataclass(slots=True)
class _OopzQwenLiveRuntimeHarness:
    """Own one shared OOPZ bus, paid Provider, and optional real music stream."""

    bot: OopzBot
    channel_sessions: OopzVoiceChannelSessionManager
    lease: OopzVoiceParticipant
    runtime: RealtimeVoiceSessionRuntimeImpl
    sessions: FakeVoiceSessionRepository
    providers: list[QwenOmniRealtimeProvider]
    audio_settings: AudioMixerSettings
    voice_channel: VoiceChannelKey
    music: OopzMusicVoiceGateway | None = None
    playback: MusicPlayback | None = None

    @classmethod
    async def create(
        cls,
        config: QwenOmniConfig,
        *,
        area_id: str,
        channel_id: str,
        target_person_id: str,
        music_stream_url: str | None = None,
        music_first: bool = False,
    ) -> _OopzQwenLiveRuntimeHarness:
        bot = OopzBot(await OopzConfig.from_env_async())
        audio_values = dict(os.environ)
        audio_values["CYWL_AUDIO_MIXER_ENABLED"] = "true"
        audio_settings = AudioMixerSettings.from_mapping(audio_values)
        master_factory = OopzMasterPcmOutputFactory.from_settings(bot, audio_settings)
        channel_sessions = OopzVoiceChannelSessionManager(
            bot,
            allow_mixed_participants=True,
            master_factory=master_factory,
            master_target_buffer_ms=audio_settings.master_target_buffer_ms,
            music_queue_ms=audio_settings.music_queue_ms,
            voice_queue_ms=audio_settings.voice_queue_ms,
            mixer_levels=audio_settings.mixer_levels(),
        )
        lease = None
        runtime = None
        music = None
        playback = None
        try:
            await bot.rest.start()
            await bot.voice.start()
            music_channel = MusicVoiceChannelKey(area_id, channel_id)
            if music_stream_url is not None and music_first:
                music, playback = await cls._start_music(
                    bot,
                    channel_sessions,
                    audio_settings,
                    music_channel,
                    music_stream_url,
                )
            lease = await channel_sessions.try_acquire(
                VoiceParticipantRequest(
                    VoiceParticipantKind.CONVERSATION,
                    AudioChannelKey(area_id, channel_id),
                    "qwen-oopz-live-e2e",
                )
            )
            if lease is None:
                pytest.fail("OOPZ voice backend is already leased")
            channel = VoiceChannelKey(area_id, channel_id)
            configuration = await FakeVoiceConfigurationRepository().resolve_start_configuration(
                target_person_id,
                channel,
            )
            descriptor = VoiceSessionDescriptor(
                uuid4(),
                target_person_id,
                channel,
                VoiceTextAddress(area_id, channel_id),
            )
            settings = VoiceSettings.from_mapping(
                {
                    "CYWL_VOICE_ENABLED": "true",
                    "CYWL_VOICE_START_TIMEOUT_SECONDS": "30",
                    "CYWL_VOICE_STOP_TIMEOUT_SECONDS": "1.5",
                    "CYWL_VOICE_IDLE_TIMEOUT_SECONDS": "300",
                    "CYWL_VOICE_MAX_SESSION_SECONDS": "1800",
                    "CYWL_VOICE_INPUT_QUEUE_MS": "1000",
                    "CYWL_VOICE_OUTPUT_QUEUE_MS": "2000",
                }
            )
            sessions = FakeVoiceSessionRepository()
            providers: list[QwenOmniRealtimeProvider] = []

            def build_provider(context):
                del context
                provider = QwenOmniRealtimeProvider(
                    config,
                    "你正在执行实时语音端到端测试。听清用户后，每次只用一句简短中文回答。",
                )
                providers.append(provider)
                return provider

            runtime = RealtimeVoiceSessionRuntimeImpl(
                VoiceSessionRuntimeContext(descriptor, lease, configuration),
                settings,
                OopzVoiceMediaGateway(
                    bot,
                    settings,
                    audio_settings,
                    master_factory=master_factory,
                ),
                sessions,
                build_provider,
            )
            await runtime.start()
            if music_stream_url is not None and not music_first:
                music, playback = await cls._start_music(
                    bot,
                    channel_sessions,
                    audio_settings,
                    music_channel,
                    music_stream_url,
                )
            return cls(
                bot,
                channel_sessions,
                lease,
                runtime,
                sessions,
                providers,
                audio_settings,
                channel,
                music,
                playback,
            )
        except BaseException:
            if runtime is not None:
                await runtime.aclose()
            if music is not None:
                await music.aclose()
            if lease is not None:
                await lease.release()
            await channel_sessions.aclose()
            try:
                await bot.voice.close()
            finally:
                await bot.rest.close()
            raise

    @staticmethod
    async def _start_music(
        bot: OopzBot,
        channel_sessions: OopzVoiceChannelSessionManager,
        audio_settings: AudioMixerSettings,
        channel: MusicVoiceChannelKey,
        stream_url: str,
    ) -> tuple[OopzMusicVoiceGateway, MusicPlayback]:
        gateway = OopzMusicVoiceGateway(bot, channel_sessions, audio_settings)
        try:
            await gateway.validate_capabilities()
            if not await gateway.acquire(channel):
                pytest.fail("shared OOPZ voice channel rejected the music participant")
            playback = await gateway.start_playback(channel, stream_url)
        except BaseException:
            await gateway.aclose()
            raise
        return gateway, playback

    async def wait_for_responses(self, count: int, timeout_seconds: float) -> None:
        finished = asyncio.create_task(self.runtime.wait_finished())
        try:
            async with asyncio.timeout(timeout_seconds):
                while self.runtime.stats.responses_drained < count:
                    if finished.done():
                        result = finished.result()
                        pytest.fail(
                            "voice runtime finished before the requested responses: "
                            f"reason={result.reason.value}"
                        )
                    await asyncio.sleep(0.05)
        finally:
            if not finished.done():
                finished.cancel()
            await asyncio.gather(finished, return_exceptions=True)

    async def stop(self) -> None:
        await self.runtime.request_stop(VoiceStopReason.COMMAND)
        async with asyncio.timeout(2):
            result = await self.runtime.wait_finished()
        assert result.reason is VoiceStopReason.COMMAND

    async def release_voice(self) -> None:
        await self.stop()
        await self.lease.release()

    async def release_music(self) -> None:
        if self.music is None:
            return
        await self.music.aclose()
        self.music = None

    async def restart_music(self, stream_url: str) -> MusicPlayback:
        if self.music is None:
            raise RuntimeError("music gateway is not active")
        self.playback = await self.music.start_playback(
            MusicVoiceChannelKey(
                self.voice_channel.area_id,
                self.voice_channel.channel_id,
            ),
            stream_url,
        )
        return self.playback

    async def aclose(self) -> None:
        try:
            await self.runtime.aclose()
        finally:
            try:
                if self.music is not None:
                    await self.music.aclose()
                    self.music = None
            finally:
                try:
                    try:
                        await self.lease.release()
                    finally:
                        await self.channel_sessions.aclose()
                finally:
                    try:
                        await self.bot.voice.close()
                    finally:
                        await self.bot.rest.close()


def test_live_wav_replay_loads_bounded_pcm_fixture(tmp_path: Path) -> None:
    path = tmp_path / "speech.wav"
    samples = np.zeros(3_200, dtype="<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(samples.tobytes())

    replay = _LiveWavReplay.load(path)

    assert replay.format == VoiceAudioFormat(16_000, 1, "f32le")
    assert len(replay.chunks) == 4
    assert sum(len(chunk) for chunk in replay.chunks) == 3_200 * 4


def test_qwen_live_counts_are_bounded(monkeypatch) -> None:
    monkeypatch.setenv("CYWL_QWEN_LIVE_ROUNDS", "20")
    assert _bounded_live_integer("CYWL_QWEN_LIVE_ROUNDS", 10, 20) == 20

    monkeypatch.setenv("CYWL_QWEN_LIVE_BARGE_INS", "0")
    with pytest.raises(ValueError, match="between 1 and 20"):
        _bounded_live_integer("CYWL_QWEN_LIVE_BARGE_INS", 20, 20)

    monkeypatch.setenv("CYWL_VOICE_E2E_ROUNDS", "11")
    with pytest.raises(ValueError, match="between 1 and 10"):
        _bounded_live_integer("CYWL_VOICE_E2E_ROUNDS", 3, 10)

    monkeypatch.setenv("CYWL_VOICE_MUSIC_E2E_ROUNDS", "0")
    with pytest.raises(ValueError, match="between 1 and 10"):
        _bounded_live_integer("CYWL_VOICE_MUSIC_E2E_ROUNDS", 10, 10)

    monkeypatch.setenv("CYWL_VOICE_MUSIC_SOAK_SECONDS", "59")
    with pytest.raises(ValueError, match="between 60 and 1800"):
        _bounded_live_soak_seconds()


def _required_live_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.fail(f"live voice E2E test requires {name}")
    return value


async def _resolve_live_music_from(
    catalog: NeteaseMusicCatalog,
    settings: MusicSettings,
) -> PlayableTrack:
    query = os.getenv("CYWL_MUSIC_LIVE_QUERY", "初音未来").strip() or "初音未来"
    tracks = await catalog.search(query, limit=settings.search_limit)
    for track in tracks:
        try:
            return await catalog.resolve(track)
        except MusicNotFoundError:
            continue
    pytest.fail("none of the bounded Netease search results is currently playable")


async def _resolve_live_music() -> PlayableTrack:
    settings = MusicSettings.from_mapping(os.environ)
    catalog = NeteaseMusicCatalog(settings)
    try:
        return await _resolve_live_music_from(catalog, settings)
    finally:
        await catalog.aclose()


async def _load_live_config() -> QwenOmniConfig:
    env_names = ("CYWL_QWEN_ENDPOINT", "CYWL_QWEN_API_KEY", "CYWL_QWEN_MODEL")
    if all(os.getenv(name) for name in env_names):
        return QwenOmniConfig(
            endpoint=os.environ["CYWL_QWEN_ENDPOINT"],
            api_key=os.environ["CYWL_QWEN_API_KEY"],
            model=os.environ["CYWL_QWEN_MODEL"],
            voice=os.getenv("CYWL_QWEN_VOICE", "Cherry"),
        )

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("Qwen live smoke requires complete CYWL_QWEN_* values or DATABASE_URL")
    engine = create_async_engine(normalize_asyncpg_url(database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            row = (
                await session.execute(
                    select(VoiceModelRecord, VoiceProviderRecord)
                    .join(
                        VoiceProviderRecord,
                        VoiceProviderRecord.id == VoiceModelRecord.provider_id,
                    )
                    .where(
                        VoiceProviderRecord.protocol == VoiceProviderProtocol.QWEN_OMNI_REALTIME_WS,
                        VoiceProviderRecord.enabled.is_(True),
                        VoiceModelRecord.enabled.is_(True),
                        VoiceModelRecord.is_application_default.is_(True),
                    )
                )
            ).one_or_none()
            if row is None:
                pytest.skip("database has no enabled application-default Qwen Omni voice model")
            model, provider = row
            credentials = dict(provider.credentials)
            if api_key := os.getenv("CYWL_QWEN_API_KEY", "").strip():
                credentials["api_key"] = api_key
            configuration = VoiceStartConfiguration(
                provider=VoiceProviderConfiguration(
                    provider.id,
                    provider.alias,
                    provider.display_name,
                    provider.protocol,
                    os.getenv("CYWL_QWEN_ENDPOINT", "").strip() or provider.endpoint,
                    MappingProxyType(credentials),
                    MappingProxyType(dict(provider.config)),
                ),
                model=VoiceModelConfiguration(
                    model.id,
                    model.provider_id,
                    model.alias,
                    os.getenv("CYWL_QWEN_MODEL", "").strip() or model.remote_model_name,
                    model.display_name,
                    model.mode,
                    MappingProxyType(dict(model.capabilities)),
                    MappingProxyType(dict(model.audio_config)),
                    MappingProxyType(dict(model.prompt_config)),
                    MappingProxyType(dict(model.limits)),
                ),
                channel=VoiceChannelConfiguration(
                    VoiceChannelKey("live-test-area", "live-test-voice"),
                    "voice_readonly_v1",
                    300,
                ),
                voice_id=os.getenv("CYWL_QWEN_VOICE", ""),
                duplex_mode=VoiceDuplexMode.FULL,
                delegated_agent_model_id=None,
            )
            try:
                return QwenOmniConfig.from_start_configuration(configuration)
            except VoiceProviderConfigurationError as exc:
                pytest.skip(f"database Qwen Omni configuration is incomplete: {exc}")
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_qwen_live_handshake_and_clean_finish() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if os.getenv("CYWL_RUN_LIVE_QWEN_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_QWEN_TESTS=1 to run the Qwen realtime smoke")
    config = await _load_live_config()
    provider = QwenOmniRealtimeProvider(
        config,
        "你正在执行连接测试。不要主动说话。",
    )
    descriptor = VoiceSessionDescriptor(
        uuid4(),
        "live-test-person",
        VoiceChannelKey("live-test-area", "live-test-voice"),
        VoiceTextAddress("live-test-area", "live-test-text"),
    )
    session = await provider.connect(descriptor)
    events = session.events()
    try:
        async with asyncio.timeout(15):
            ready = await anext(events)
            assert isinstance(ready, VoiceSessionReady)
            await session.finish()
            terminal = await anext(events)
            assert isinstance(terminal, VoiceSessionFinished)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_qwen_live_runtime_replays_multiple_audio_turns() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if os.getenv("CYWL_RUN_LIVE_QWEN_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_QWEN_TESTS=1 to run the Qwen realtime smoke")
    wav_path = os.getenv("CYWL_QWEN_LIVE_INPUT_WAV", "").strip()
    if not wav_path:
        pytest.skip("set CYWL_QWEN_LIVE_INPUT_WAV to a local spoken PCM WAV")
    rounds = _bounded_live_integer("CYWL_QWEN_LIVE_ROUNDS", 10, 20)
    replay = _LiveWavReplay.load(Path(wav_path))
    config = await _load_live_config()
    harness = await _QwenLiveRuntimeHarness.create(
        config,
        "你正在执行实时语音多轮测试。听清用户后，每次只用一句简短中文回答。",
    )
    try:
        sequence = 0
        for expected in range(1, rounds + 1):
            sequence = await replay.replay(
                harness.media,
                sequence,
                trailing_silence_ms=1_200,
            )
            async with asyncio.timeout(60):
                while harness.runtime.stats.responses_drained < expected:
                    await asyncio.sleep(0.05)

        user_turns = sum(turn[2] is VoiceTurnRole.USER for turn in harness.sessions.turns)
        assistant_turns = sum(turn[2] is VoiceTurnRole.ASSISTANT for turn in harness.sessions.turns)
        assert harness.runtime.stats.responses_started >= rounds
        assert harness.runtime.stats.responses_drained >= rounds
        assert harness.runtime.stats.first_final_transcript_ms > 0
        assert harness.runtime.stats.first_provider_audio_ms > 0
        assert harness.runtime.stats.first_oopz_output_ms > 0
        assert len(harness.media.outputs) >= rounds
        assert user_turns >= rounds
        assert assistant_turns >= rounds

        await harness.stop()
    finally:
        await harness.aclose()
    assert harness.media.closed is True
    assert all(provider._closed for provider in harness.providers)


@pytest.mark.asyncio
async def test_qwen_live_runtime_survives_repeated_barge_in() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if os.getenv("CYWL_RUN_LIVE_QWEN_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_QWEN_TESTS=1 to run the Qwen realtime smoke")
    wav_path = os.getenv("CYWL_QWEN_LIVE_INPUT_WAV", "").strip()
    if not wav_path:
        pytest.skip("set CYWL_QWEN_LIVE_INPUT_WAV to a local spoken PCM WAV")
    barge_ins = _bounded_live_integer("CYWL_QWEN_LIVE_BARGE_INS", 20, 20)
    replay = _LiveWavReplay.load(Path(wav_path))
    config = await _load_live_config()
    harness = await _QwenLiveRuntimeHarness.create(
        config,
        (
            "你正在执行实时语音打断测试。每次听到用户后，用中文从一数到一百，"
            "保持自然语速并持续说至少十秒；被打断后立刻停止，再对新的话重新开始。"
        ),
    )
    harness.media.drain_gate = asyncio.Event()
    try:
        sequence = await replay.replay(harness.media, 0, trailing_silence_ms=1_000)
        async with asyncio.timeout(60):
            while harness.runtime.state is not VoiceSessionState.SPEAKING:
                await asyncio.sleep(0.01)

        for expected in range(1, barge_ins + 1):
            sequence = await replay.replay(
                harness.media,
                sequence,
                trailing_silence_ms=1_000,
            )
            async with asyncio.timeout(60):
                while harness.runtime.stats.barge_in_count < expected:
                    await asyncio.sleep(0.01)
            if expected < barge_ins:
                async with asyncio.timeout(60):
                    while not (
                        harness.runtime.stats.responses_started >= expected + 1
                        and harness.runtime.state is VoiceSessionState.SPEAKING
                    ):
                        await asyncio.sleep(0.01)

        assert harness.runtime.stats.barge_in_count == barge_ins
        assert len(harness.media.flushes) >= barge_ins
        assert harness.runtime.stats.max_barge_in_flush_ms < 200
        logger.info(
            "Qwen live barge-in gate: count=%s max_local_flush_ms=%.1f responses=%s",
            barge_ins,
            harness.runtime.stats.max_barge_in_flush_ms,
            harness.runtime.stats.responses_started,
        )
        harness.media.drain_gate.set()
        await harness.stop()
    finally:
        await harness.aclose()
    assert harness.media.closed is True
    assert all(provider._closed for provider in harness.providers)


@pytest.mark.asyncio
async def test_qwen_live_runtime_closes_provider_on_idle_timeout() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if os.getenv("CYWL_RUN_LIVE_QWEN_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_QWEN_TESTS=1 to run the Qwen realtime smoke")
    config = await _load_live_config()
    harness = await _QwenLiveRuntimeHarness.create(
        config,
        "你正在执行空闲超时测试。不要主动说话。",
        idle_timeout_seconds=1,
    )
    try:
        async with asyncio.timeout(3):
            result = await harness.runtime.wait_finished()
        assert result.reason is VoiceStopReason.IDLE_TIMEOUT
    finally:
        await harness.aclose()
    assert harness.media.closed is True
    assert all(provider._closed for provider in harness.providers)


@pytest.mark.asyncio
async def test_qwen_live_runtime_closes_provider_when_owner_leaves() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if os.getenv("CYWL_RUN_LIVE_QWEN_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_QWEN_TESTS=1 to run the Qwen realtime smoke")
    config = await _load_live_config()
    harness = await _QwenLiveRuntimeHarness.create(
        config,
        "你正在执行离开频道测试。不要主动说话。",
        owner_leave_grace_seconds=0,
    )
    try:
        await harness.media.end_input(VoiceMediaEndReason.OWNER_LEFT)
        async with asyncio.timeout(2):
            result = await harness.runtime.wait_finished()
        assert result.reason is VoiceStopReason.OWNER_LEFT
    finally:
        await harness.aclose()
    assert harness.media.closed is True
    assert all(provider._closed for provider in harness.providers)


@pytest.mark.asyncio
async def test_qwen_live_runtime_traverses_real_oopz_media_end_to_end() -> None:
    """Manually speak in the target room and hear each response before this returns."""

    load_dotenv(find_dotenv(usecwd=True), override=False)
    if os.getenv("CYWL_RUN_LIVE_VOICE_E2E_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_VOICE_E2E_TESTS=1 for explicit paid RTC mutation")
    config = await _load_live_config()
    rounds = _bounded_live_integer("CYWL_VOICE_E2E_ROUNDS", 3, 10)
    harness = await _OopzQwenLiveRuntimeHarness.create(
        config,
        area_id=_required_live_value("OOPZ_AREA_ID"),
        channel_id=_required_live_value("OOPZ_CHANNEL_ID"),
        target_person_id=_required_live_value("OOPZ_TARGET_PERSON_UID"),
    )
    try:
        await harness.wait_for_responses(rounds, timeout_seconds=rounds * 90)
        stats = harness.runtime.stats
        user_turns = sum(turn[2] is VoiceTurnRole.USER for turn in harness.sessions.turns)
        assistant_turns = sum(turn[2] is VoiceTurnRole.ASSISTANT for turn in harness.sessions.turns)
        assert stats.responses_started >= rounds
        assert stats.responses_drained >= rounds
        assert stats.first_final_transcript_ms > 0
        assert stats.first_provider_audio_ms > 0
        assert stats.first_oopz_output_ms > 0
        assert user_turns >= rounds
        assert assistant_turns >= rounds
        logger.info(
            "OOPZ/Qwen E2E gate: responses=%s first_transcript_ms=%.1f "
            "first_provider_audio_ms=%.1f first_oopz_output_ms=%.1f "
            "input_depth=%s output_depth=%s provider_reconnects=%s media_reconnects=%s",
            stats.responses_drained,
            stats.first_final_transcript_ms,
            stats.first_provider_audio_ms,
            stats.first_oopz_output_ms,
            stats.max_input_queue_depth,
            stats.max_output_queue_depth,
            stats.provider_reconnects,
            stats.media_reconnects,
        )
        await harness.stop()
    finally:
        await harness.aclose()
    assert all(provider._closed for provider in harness.providers)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("music_first", "release_voice_first"),
    ((True, True), (False, False)),
    ids=("music-first-voice-stops", "voice-first-music-stops"),
)
async def test_live_oopz_qwen_and_netease_share_one_master_bus(
    music_first: bool,
    release_voice_first: bool,
) -> None:
    """Speak for every round while the resolved Netease track remains audible."""

    load_dotenv(find_dotenv(usecwd=True), override=False)
    if os.getenv("CYWL_RUN_LIVE_VOICE_MUSIC_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_VOICE_MUSIC_TESTS=1 for paid RTC/music mutation")
    config = await _load_live_config()
    playable = await _resolve_live_music()
    rounds = _bounded_live_integer("CYWL_VOICE_MUSIC_E2E_ROUNDS", 10, 10)
    harness = await _OopzQwenLiveRuntimeHarness.create(
        config,
        area_id=_required_live_value("OOPZ_AREA_ID"),
        channel_id=_required_live_value("OOPZ_CHANNEL_ID"),
        target_person_id=_required_live_value("OOPZ_TARGET_PERSON_UID"),
        music_stream_url=playable.stream_url,
        music_first=music_first,
    )
    assert harness.playback is not None
    playback_finished = asyncio.create_task(harness.playback.wait_finished())
    try:
        snapshot = await harness.channel_sessions.current()
        assert snapshot is not None
        assert {participant.kind for participant in snapshot.participants} == {
            VoiceParticipantKind.MUSIC,
            VoiceParticipantKind.CONVERSATION,
        }

        await harness.wait_for_responses(rounds, timeout_seconds=rounds * 90)
        if playback_finished.done():
            result = playback_finished.result()
            pytest.fail(f"music ended during voice rounds: reason={result.end_reason.value}")
        bus = await harness.lease.audio_bus()
        stats = await bus.stats()
        runtime_stats = harness.runtime.stats
        logger.info(
            "OOPZ/Qwen/Netease shared gate: order=%s responses=%s metrics=%s",
            "music_first" if music_first else "voice_first",
            runtime_stats.responses_drained,
            stats.as_metrics(),
        )
        user_turns = sum(turn[2] is VoiceTurnRole.USER for turn in harness.sessions.turns)
        assistant_turns = sum(turn[2] is VoiceTurnRole.ASSISTANT for turn in harness.sessions.turns)
        assert runtime_stats.responses_started >= rounds
        assert runtime_stats.responses_drained >= rounds
        assert runtime_stats.first_final_transcript_ms > 0
        assert runtime_stats.first_provider_audio_ms > 0
        assert runtime_stats.first_oopz_output_ms > 0
        assert user_turns >= rounds
        assert assistant_turns >= rounds
        assert stats.decoder_start_ms > 0
        assert stats.master_max_buffered_ms <= harness.audio_settings.master_max_buffer_ms
        assert stats.hard_clip_samples == 0

        if release_voice_first:
            await harness.release_voice()
            snapshot = await harness.channel_sessions.current()
            assert snapshot is not None
            assert {participant.kind for participant in snapshot.participants} == {
                VoiceParticipantKind.MUSIC
            }
            await asyncio.sleep(0.2)
            assert not playback_finished.done()
        else:
            await harness.release_music()
            snapshot = await harness.channel_sessions.current()
            assert snapshot is not None
            assert {participant.kind for participant in snapshot.participants} == {
                VoiceParticipantKind.CONVERSATION
            }
            assert harness.runtime.state not in {
                VoiceSessionState.CLOSING,
                VoiceSessionState.CLOSED,
                VoiceSessionState.FAILED,
            }
            await harness.stop()
    finally:
        if not playback_finished.done():
            playback_finished.cancel()
        await asyncio.gather(playback_finished, return_exceptions=True)
        await harness.aclose()
    assert all(provider._closed for provider in harness.providers)


@pytest.mark.asyncio
async def test_live_oopz_qwen_netease_wall_clock_soak() -> None:
    """Keep speaking intermittently while music is re-resolved and looped."""

    load_dotenv(find_dotenv(usecwd=True), override=False)
    if os.getenv("CYWL_RUN_LIVE_VOICE_MUSIC_SOAK_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_VOICE_MUSIC_SOAK_TESTS=1 for the wall-clock RTC soak")
    duration_seconds = _bounded_live_soak_seconds()
    minimum_responses = _bounded_live_integer(
        "CYWL_VOICE_MUSIC_SOAK_MIN_RESPONSES",
        10,
        100,
    )
    if duration_seconds < 1_800:
        logger.warning(
            "Running shortened voice/music RTC soak; this does not satisfy Phase G: "
            "duration_seconds=%s",
            duration_seconds,
        )
    config = await _load_live_config()
    music_settings = MusicSettings.from_mapping(os.environ)
    catalog = NeteaseMusicCatalog(music_settings)
    try:
        playable = await _resolve_live_music_from(catalog, music_settings)
        harness = await _OopzQwenLiveRuntimeHarness.create(
            config,
            area_id=_required_live_value("OOPZ_AREA_ID"),
            channel_id=_required_live_value("OOPZ_CHANNEL_ID"),
            target_person_id=_required_live_value("OOPZ_TARGET_PERSON_UID"),
            music_stream_url=playable.stream_url,
            music_first=True,
        )
    except BaseException:
        await catalog.aclose()
        raise
    assert harness.playback is not None
    bus = await harness.lease.audio_bus()
    playback_finished = asyncio.create_task(harness.playback.wait_finished())
    runtime_finished = asyncio.create_task(harness.runtime.wait_finished())
    buffered_samples: list[float] = []
    track_restarts = 0
    started_at = time.monotonic()
    try:
        deadline = asyncio.get_running_loop().time() + duration_seconds
        while (remaining := deadline - asyncio.get_running_loop().time()) > 0:
            if runtime_finished.done():
                result = runtime_finished.result()
                pytest.fail(f"voice runtime ended during soak: reason={result.reason.value}")
            if playback_finished.done():
                result = playback_finished.result()
                if result.end_reason is not MusicPlaybackEndReason.FINISHED:
                    error = (
                        type(result.terminal_error).__name__
                        if result.terminal_error is not None
                        else "none"
                    )
                    pytest.fail(
                        f"music failed during soak: reason={result.end_reason.value} error={error}"
                    )
                try:
                    playable = await catalog.resolve(playable.track)
                except MusicNotFoundError:
                    playable = await _resolve_live_music_from(catalog, music_settings)
                playback = await harness.restart_music(playable.stream_url)
                playback_finished = asyncio.create_task(playback.wait_finished())
                track_restarts += 1
            stats = await bus.stats()
            buffered_samples.append(stats.master_buffered_ms)
            assert stats.master_max_buffered_ms <= harness.audio_settings.master_max_buffer_ms
            assert stats.hard_clip_samples == 0
            await asyncio.sleep(min(1.0, remaining))

        runtime_stats = harness.runtime.stats
        if runtime_stats.responses_drained < minimum_responses:
            pytest.fail(
                "not enough manual voice turns during soak: "
                f"actual={runtime_stats.responses_drained} required={minimum_responses}"
            )
        p95_index = max(0, math.ceil(len(buffered_samples) * 0.95) - 1)
        buffered_p95_ms = sorted(buffered_samples)[p95_index]
        stats = await bus.stats()
        elapsed_seconds = time.monotonic() - started_at
        logger.info(
            "OOPZ/Qwen/Netease wall-clock soak: elapsed_s=%.1f responses=%s "
            "track_restarts=%s master_buffered_p95_ms=%.1f metrics=%s",
            elapsed_seconds,
            runtime_stats.responses_drained,
            track_restarts,
            buffered_p95_ms,
            stats.as_metrics(),
        )
        assert elapsed_seconds >= duration_seconds
        assert 40 <= buffered_p95_ms <= 80
        assert stats.decoder_start_ms > 0
        assert stats.hard_clip_samples == 0

        await harness.release_voice()
        snapshot = await harness.channel_sessions.current()
        assert snapshot is not None
        assert {participant.kind for participant in snapshot.participants} == {
            VoiceParticipantKind.MUSIC
        }
    finally:
        for task in (playback_finished, runtime_finished):
            if not task.done():
                task.cancel()
        await asyncio.gather(playback_finished, runtime_finished, return_exceptions=True)
        try:
            await harness.aclose()
        finally:
            await catalog.aclose()

    assert await harness.channel_sessions.current() is None
    assert all(provider._closed for provider in harness.providers)
    final_stats = await bus.stats()
    assert final_stats.retained_source_count == 0
    assert final_stats.ledger_entry_count == 0
