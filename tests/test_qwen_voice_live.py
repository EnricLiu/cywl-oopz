from __future__ import annotations

import asyncio
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import numpy as np
import pytest
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.voice.errors import VoiceProviderConfigurationError
from cywl_oopz.features.voice.events import VoiceSessionFinished, VoiceSessionReady
from cywl_oopz.features.voice.models import (
    RemoteAudioFrame,
    VoiceAudioFormat,
    VoiceChannelKey,
    VoiceSessionDescriptor,
    VoiceStopReason,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.ports import VoiceSessionRuntimeContext
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
from cywl_oopz.integrations.voice.fake import (
    FakeVoiceAccessGateway,
    FakeVoiceConfigurationRepository,
    FakeVoiceMediaGateway,
    FakeVoiceSessionRepository,
)
from cywl_oopz.integrations.voice.qwen_omni import QwenOmniRealtimeProvider
from cywl_oopz.integrations.voice.qwen_protocol import QwenOmniConfig
from cywl_oopz.settings import VoiceSettings
from cywl_oopz.storage.models import VoiceModelRecord, VoiceProviderRecord
from cywl_oopz.storage.url import normalize_asyncpg_url

_LIVE_FRAME_SAMPLES = 1_024


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
            configuration = VoiceStartConfiguration(
                provider=VoiceProviderConfiguration(
                    provider.id,
                    provider.alias,
                    provider.display_name,
                    provider.protocol,
                    provider.endpoint,
                    MappingProxyType(dict(provider.credentials)),
                    MappingProxyType(dict(provider.config)),
                ),
                model=VoiceModelConfiguration(
                    model.id,
                    model.provider_id,
                    model.alias,
                    model.remote_model_name,
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
                pytest.skip(f"database Qwen Omni configuration is incomplete: {type(exc).__name__}")
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
    try:
        rounds = int(os.getenv("CYWL_QWEN_LIVE_ROUNDS", "10"))
    except ValueError:
        pytest.fail("CYWL_QWEN_LIVE_ROUNDS must be an integer")
    if not 1 <= rounds <= 20:
        pytest.fail("CYWL_QWEN_LIVE_ROUNDS must be between 1 and 20")
    replay = _LiveWavReplay.load(Path(wav_path))
    config = await _load_live_config()
    access = FakeVoiceAccessGateway()
    channel = VoiceChannelKey("live-test-area", "live-test-voice")
    lease = await access.try_acquire(channel, "qwen-live-runtime")
    assert lease is not None
    configuration = await FakeVoiceConfigurationRepository().resolve_start_configuration(
        "live-test-person", channel
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
        provider = QwenOmniRealtimeProvider(
            config,
            "你正在执行实时语音多轮测试。听清用户后，每次只用一句简短中文回答。",
        )
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
                "CYWL_VOICE_MAX_SESSION_SECONDS": "900",
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
        media = media_gateway.sessions[0]
        sequence = 0
        for expected in range(1, rounds + 1):
            sequence = await replay.replay(media, sequence, trailing_silence_ms=1_200)
            async with asyncio.timeout(60):
                while runtime.stats.responses_drained < expected:
                    await asyncio.sleep(0.05)

        user_turns = sum(turn[2] is VoiceTurnRole.USER for turn in sessions.turns)
        assistant_turns = sum(turn[2] is VoiceTurnRole.ASSISTANT for turn in sessions.turns)
        assert runtime.stats.responses_started >= rounds
        assert runtime.stats.responses_drained >= rounds
        assert runtime.stats.first_final_transcript_ms > 0
        assert runtime.stats.first_provider_audio_ms > 0
        assert runtime.stats.first_oopz_output_ms > 0
        assert len(media.outputs) >= rounds
        assert user_turns >= rounds
        assert assistant_turns >= rounds

        await runtime.request_stop(VoiceStopReason.COMMAND)
        async with asyncio.timeout(2):
            result = await runtime.wait_finished()
        assert result.reason is VoiceStopReason.COMMAND
    finally:
        await runtime.aclose()
        await lease.release()
    assert media_gateway.sessions[0].closed is True
    assert all(provider._closed for provider in providers)
