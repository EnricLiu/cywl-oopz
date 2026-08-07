from __future__ import annotations

import asyncio
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from cywl_oopz.features.audio.models import AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS
from cywl_oopz.features.music.errors import (
    MusicDecoderError,
    MusicDecoderUnavailableError,
)
from cywl_oopz.integrations.audio.ffmpeg import (
    FfmpegCapabilityProbe,
    FfmpegMusicDecoder,
    FfmpegMusicDecoderFactory,
)
from cywl_oopz.settings import AudioMixerSettings


def settings(**changes: str) -> AudioMixerSettings:
    return AudioMixerSettings.from_mapping(
        {
            "CYWL_AUDIO_MIXER_ENABLED": "true",
            "CYWL_AUDIO_DECODER_START_TIMEOUT_SECONDS": "0.2",
            "CYWL_AUDIO_DECODER_READ_TIMEOUT_SECONDS": "0.2",
            "CYWL_AUDIO_DECODER_STOP_TIMEOUT_SECONDS": "0.02",
            **changes,
        }
    )


class FakeProcess:
    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        *,
        returncode: int | None = 0,
        communicate_output: bytes = b"",
    ) -> None:
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self.returncode = returncode
        self.communicate_output = communicate_output
        self.terminate_calls = 0
        self.kill_calls = 0

    async def communicate(self):
        return self.communicate_output, None

    async def wait(self) -> int:
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def pcm(frames: int, value: float = 0.25) -> bytes:
    return np.full((frames, AUDIO_CHANNELS), value, dtype="<f4").tobytes()


@pytest.mark.asyncio
async def test_ffmpeg_capability_probe_rejects_missing_and_accepts_real_signature() -> None:
    missing = FfmpegCapabilityProbe("missing", resolver=lambda _name: None)
    with pytest.raises(MusicDecoderUnavailableError, match="not found"):
        await missing.validate()

    calls: list[tuple[object, ...]] = []

    async def process_factory(*args, **_kwargs):
        calls.append(args)
        return FakeProcess(communicate_output=b"ffmpeg version fixture\n")

    probe = FfmpegCapabilityProbe(
        "ffmpeg",
        process_factory=process_factory,
        resolver=lambda _name: "/fixture/ffmpeg",
    )

    assert await probe.validate() == "/fixture/ffmpeg"
    assert calls == [("/fixture/ffmpeg", "-version")]


def test_ffmpeg_command_is_argument_list_with_canonical_output() -> None:
    stream_url = "https://music.example/song?id=1&token=sensitive"
    command = FfmpegMusicDecoder.command("/usr/bin/ffmpeg", stream_url, settings())

    assert command[0] == "/usr/bin/ffmpeg"
    assert command[command.index("-i") + 1] == stream_url
    assert "-reconnect" in command
    assert "-reconnect_max_retries" not in command
    assert command[-9:] == (
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "pipe:1",
    )
    assert "shell=True" not in command

    local_command = FfmpegMusicDecoder.command(
        "/usr/bin/ffmpeg",
        "/tmp/music.wav",
        settings(),
    )
    assert "-rw_timeout" not in local_command
    assert "-reconnect" not in local_command


@pytest.mark.asyncio
async def test_ffmpeg_decoder_emits_full_and_zero_padded_tail() -> None:
    process = FakeProcess(
        pcm(AUDIO_BLOCK_FRAMES) + pcm(17, value=float("nan")),
        stderr=b"warning\n",
    )

    async def process_factory(*_args, **_kwargs):
        return process

    decoder = await FfmpegMusicDecoder.open(
        "/fixture/ffmpeg",
        "https://music.example/audio",
        settings(),
        process_factory=process_factory,
    )
    blocks = [block async for block in decoder]

    assert [block.valid_frames for block in blocks] == [AUDIO_BLOCK_FRAMES, 17]
    assert np.allclose(blocks[0].samples, 0.25)
    assert np.all(blocks[1].samples[:17] == 0)
    assert np.all(blocks[1].samples[17:] == 0)
    assert decoder.stats.invalid_samples == 34
    assert decoder.stats.stderr_bytes == len(b"warning\n")
    await decoder.aclose()


@pytest.mark.asyncio
async def test_ffmpeg_decoder_rejects_misaligned_tail_and_hides_stderr() -> None:
    secret = "https://music.example/audio?token=do-not-log"
    process = FakeProcess(b"abc", secret.encode(), returncode=1)

    async def process_factory(*_args, **_kwargs):
        return process

    with pytest.raises(MusicDecoderError) as failure:
        await FfmpegMusicDecoder.open(
            "/fixture/ffmpeg",
            secret,
            settings(),
            process_factory=process_factory,
        )

    assert secret not in str(failure.value)
    assert "do-not-log" not in str(failure.value)


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
async def test_real_ffmpeg_decodes_local_wav_when_available(tmp_path: Path) -> None:
    wav_path = tmp_path / "tone.wav"
    frames = 4_800
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes((1000).to_bytes(2, "little", signed=True) * frames)

    factory = FfmpegMusicDecoderFactory(settings())
    decoder = await factory.open(str(wav_path))
    decoded = [block async for block in decoder]

    assert sum(block.valid_frames for block in decoded) == frames
    await decoder.aclose()
