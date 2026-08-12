from __future__ import annotations

import asyncio
import logging
import shutil
import wave
from collections import deque
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from cywl_oopz.features.audio.models import AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS
from cywl_oopz.features.music.errors import (
    MusicDecoderError,
    MusicDecoderUnavailableError,
)
from cywl_oopz.features.music.models import ResolvedMediaInput
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


def media(url: str) -> ResolvedMediaInput:
    return ResolvedMediaInput(url)


class FakeProcess:
    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        *,
        returncode: int | None = 0,
        communicate_output: bytes = b"",
    ) -> None:
        self.stdout = RecordingStreamReader()
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


class RecordingStreamReader(asyncio.StreamReader):
    def __init__(self) -> None:
        super().__init__()
        self.bytes_read = 0

    async def read(self, n: int = -1) -> bytes:
        payload = await super().read(n)
        self.bytes_read += len(payload)
        return payload


def pcm(frames: int, value: float = 0.25) -> bytes:
    return np.full((frames, AUDIO_CHANNELS), value, dtype="<f4").tobytes()


def wav_bytes(frames: int) -> bytes:
    payload = BytesIO()
    with wave.open(payload, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes((1000).to_bytes(2, "little", signed=True) * frames)
    return payload.getvalue()


class LocalAudioHttpServer:
    """Serve deterministic HTTP outcomes without external network dependencies."""

    def __init__(self, responses: tuple[tuple[int, bytes | None], ...]) -> None:
        if not responses:
            raise ValueError("Local HTTP fixture requires at least one response")
        self._responses = deque(responses)
        self._last_response = responses[-1]
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._closing = asyncio.Event()
        self.requests = 0
        self.request_headers: list[dict[str, str]] = []
        self.port = 0

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        socket = self._server.sockets[0]
        self.port = int(socket.getsockname()[1])

    def url(self, query: str = "") -> str:
        suffix = f"?{query}" if query else ""
        return f"http://127.0.0.1:{self.port}/audio.wav{suffix}"

    async def aclose(self) -> None:
        self._closing.set()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for writer in tuple(self._writers):
            writer.close()
        await asyncio.gather(
            *(writer.wait_closed() for writer in tuple(self._writers)),
            return_exceptions=True,
        )
        await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.create_task(self._serve(reader, writer), name="audio-http-fixture")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _serve(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._writers.add(writer)
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            headers: dict[str, str] = {}
            for line in request.decode("latin-1").split("\r\n")[1:]:
                name, separator, value = line.partition(":")
                if separator:
                    headers[name.strip().casefold()] = value.strip()
            self.request_headers.append(headers)
            self.requests += 1
            status, body = self._responses.popleft() if self._responses else self._last_response
            if status == 503:
                writer.write(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Length: 0\r\n"
                    b"Retry-After: 0\r\n"
                    b"Connection: close\r\n\r\n"
                )
                await writer.drain()
                return
            content_length = len(body) if body is not None else 1_000_000
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: audio/wav\r\n"
                + f"Content-Length: {content_length}\r\n".encode()
                + b"Connection: close\r\n\r\n"
            )
            if body is None:
                await writer.drain()
                await self._closing.wait()
            else:
                writer.write(body)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self._writers.discard(writer)
            writer.close()
            await asyncio.gather(writer.wait_closed(), return_exceptions=True)


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
    command = FfmpegMusicDecoder.command("/usr/bin/ffmpeg", media(stream_url), settings())

    assert command[0] == "/usr/bin/ffmpeg"
    assert command[command.index("-i") + 1] == stream_url
    assert "-reconnect" in command
    assert command[command.index("-reconnect_max_retries") + 1] == "3"
    assert command[command.index("-reconnect_delay_total_max") + 1] == "10"
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
        media("/tmp/music.wav"),
        settings(),
    )
    assert "-rw_timeout" not in local_command
    assert "-reconnect" not in local_command


def test_ffmpeg_command_places_sanitized_headers_before_input() -> None:
    source = ResolvedMediaInput(
        "https://music.example/audio",
        (
            ("User-Agent", "CYWL fixture"),
            ("Referer", "https://www.bilibili.com/video/BV1"),
            ("Origin", "https://www.bilibili.com"),
            ("Accept-Language", "zh-CN"),
            ("Cookie", "not-forwarded"),
        ),
    )

    command = FfmpegMusicDecoder.command("ffmpeg", source, settings())
    input_index = command.index("-i")

    assert command.index("-user_agent") < input_index
    assert command[command.index("-user_agent") + 1] == "CYWL fixture"
    assert command.index("-referer") < input_index
    serialized = command[command.index("-headers") + 1]
    assert serialized == "Origin: https://www.bilibili.com\r\nAccept-Language: zh-CN\r\n"
    assert "not-forwarded" not in command


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
        media("https://music.example/audio"),
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
async def test_ffmpeg_decoder_pulls_one_block_without_user_space_read_ahead() -> None:
    one_block = pcm(AUDIO_BLOCK_FRAMES)
    process = FakeProcess(one_block * 20)

    async def process_factory(*_args, **_kwargs):
        return process

    decoder = await FfmpegMusicDecoder.open(
        "/fixture/ffmpeg",
        media("https://music.example/audio"),
        settings(),
        process_factory=process_factory,
    )

    assert process.stdout.bytes_read == len(one_block)
    await asyncio.sleep(0.01)
    assert process.stdout.bytes_read == len(one_block)
    await anext(decoder)
    assert process.stdout.bytes_read == len(one_block)
    await anext(decoder)
    assert process.stdout.bytes_read == len(one_block) * 2
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
            media(secret),
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
    decoder = await factory.open(media(str(wav_path)))
    decoded = [block async for block in decoder]

    assert sum(block.valid_frames for block in decoded) == frames
    await decoder.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
async def test_real_ffmpeg_closes_early_with_buffered_stdout(tmp_path: Path) -> None:
    wav_path = tmp_path / "long-tone.wav"
    frames = 48_000 * 10
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes((1000).to_bytes(2, "little", signed=True) * frames)

    factory = FfmpegMusicDecoderFactory(settings(CYWL_AUDIO_DECODER_STOP_TIMEOUT_SECONDS="0.2"))
    decoder = await factory.open(media(str(wav_path)))
    await asyncio.sleep(0.05)

    async with asyncio.timeout(1):
        await decoder.aclose()
    await decoder.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
async def test_real_ffmpeg_slow_http_startup_times_out_without_logging_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = LocalAudioHttpServer(((200, None),))
    await server.start()
    secret = "token=do-not-log"
    try:
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(MusicDecoderError, match="startup timed out"):
                await FfmpegMusicDecoder.open(
                    shutil.which("ffmpeg") or "ffmpeg",
                    media(server.url(secret)),
                    settings(
                        CYWL_AUDIO_DECODER_START_TIMEOUT_SECONDS="2",
                        CYWL_AUDIO_DECODER_READ_TIMEOUT_SECONDS="5",
                        CYWL_AUDIO_DECODER_STOP_TIMEOUT_SECONDS="0.2",
                    ),
                )
    finally:
        await server.aclose()

    assert server.requests >= 1
    assert secret not in caplog.text
    assert "do-not-log" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
async def test_real_ffmpeg_reconnects_after_temporary_http_503() -> None:
    frames = 4_800
    server = LocalAudioHttpServer(((503, b""), (200, wav_bytes(frames))))
    await server.start()
    decoder: FfmpegMusicDecoder | None = None
    try:
        decoder = await FfmpegMusicDecoder.open(
            shutil.which("ffmpeg") or "ffmpeg",
            media(server.url()),
            settings(
                CYWL_AUDIO_DECODER_START_TIMEOUT_SECONDS="5",
                CYWL_AUDIO_DECODER_READ_TIMEOUT_SECONDS="2",
                CYWL_AUDIO_DECODER_STOP_TIMEOUT_SECONDS="0.5",
            ),
        )
        decoded = [block async for block in decoder]
    finally:
        if decoder is not None:
            await decoder.aclose()
        await server.aclose()

    assert server.requests >= 2
    assert sum(block.valid_frames for block in decoded) == frames


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
async def test_real_ffmpeg_forwards_provider_media_headers() -> None:
    frames = 4_800
    server = LocalAudioHttpServer(((200, wav_bytes(frames)),))
    await server.start()
    decoder: FfmpegMusicDecoder | None = None
    try:
        decoder = await FfmpegMusicDecoder.open(
            shutil.which("ffmpeg") or "ffmpeg",
            ResolvedMediaInput(
                server.url(),
                (
                    ("User-Agent", "CYWL header fixture"),
                    ("Referer", "https://www.bilibili.com/video/BV1fixture"),
                    ("Origin", "https://www.bilibili.com"),
                ),
            ),
            settings(
                CYWL_AUDIO_DECODER_START_TIMEOUT_SECONDS="5",
                CYWL_AUDIO_DECODER_READ_TIMEOUT_SECONDS="2",
            ),
        )
        decoded = [block async for block in decoder]
    finally:
        if decoder is not None:
            await decoder.aclose()
        await server.aclose()

    assert sum(block.valid_frames for block in decoded) == frames
    assert server.request_headers[0]["user-agent"] == "CYWL header fixture"
    assert server.request_headers[0]["referer"] == ("https://www.bilibili.com/video/BV1fixture")
    assert server.request_headers[0]["origin"] == "https://www.bilibili.com"
