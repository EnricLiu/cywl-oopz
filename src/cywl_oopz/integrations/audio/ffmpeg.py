"""Supervised FFmpeg decoding into canonical float32 stereo blocks."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from urllib.parse import urlparse

import numpy as np

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.audio.models import (
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    CANONICAL_AUDIO_FORMAT,
    DecodedAudioBlock,
)
from cywl_oopz.features.music.errors import (
    MusicDecoderError,
    MusicDecoderUnavailableError,
)
from cywl_oopz.settings import AudioMixerSettings

logger = logging.getLogger(__name__)

_STDERR_TAIL_BYTES = 16 * 1024
_BLOCK_BYTES = AUDIO_BLOCK_FRAMES * CANONICAL_AUDIO_FORMAT.frame_width_bytes
_ProcessFactory = Callable[..., Awaitable[asyncio.subprocess.Process]]
_ExecutableResolver = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class FfmpegDecoderStats:
    """Non-sensitive decoder counters for diagnostics and tests."""

    blocks_emitted: int = 0
    valid_frames_emitted: int = 0
    invalid_samples: int = 0
    stderr_bytes: int = 0
    forced_kills: int = 0


class FfmpegCapabilityProbe:
    """Resolve and execute the configured binary before accepting PCM music traffic."""

    def __init__(
        self,
        executable: str,
        *,
        timeout_seconds: float = 3.0,
        process_factory: _ProcessFactory = asyncio.create_subprocess_exec,
        resolver: _ExecutableResolver = shutil.which,
    ) -> None:
        if not executable.strip() or timeout_seconds <= 0:
            raise ValueError("FFmpeg capability probe configuration is invalid")
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._process_factory = process_factory
        self._resolver = resolver

    async def validate(self) -> str:
        resolved = self._resolve()
        if resolved is None:
            raise MusicDecoderUnavailableError("Configured FFmpeg executable was not found")
        process: asyncio.subprocess.Process | None = None
        try:
            process = await self._process_factory(
                resolved,
                "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async with asyncio.timeout(self._timeout_seconds):
                output, _ = await process.communicate()
        except asyncio.CancelledError:
            await self._stop_probe(process)
            raise
        except Exception as exc:
            await self._stop_probe(process)
            raise MusicDecoderUnavailableError(
                f"FFmpeg capability check failed: {exception_kind(exc)}"
            ) from exc
        first_line = output.splitlines()[0] if output else b""
        if process.returncode != 0 or not first_line.startswith(b"ffmpeg version"):
            raise MusicDecoderUnavailableError("Configured executable is not compatible FFmpeg")
        logger.info("FFmpeg PCM decoder capability validated")
        return resolved

    @staticmethod
    async def _stop_probe(process: asyncio.subprocess.Process | None) -> None:
        if process is None or process.returncode is not None:
            return
        process.kill()
        with suppress(BaseException):
            await asyncio.shield(process.wait())

    def _resolve(self) -> str | None:
        if os.path.dirname(self._executable):
            return self._executable if os.access(self._executable, os.X_OK) else None
        return self._resolver(self._executable)


class FfmpegMusicDecoderFactory:
    """Cache one validated executable and open independently supervised decoders."""

    def __init__(
        self,
        settings: AudioMixerSettings,
        *,
        process_factory: _ProcessFactory = asyncio.create_subprocess_exec,
        resolver: _ExecutableResolver = shutil.which,
    ) -> None:
        self._settings = settings
        self._process_factory = process_factory
        self._probe = FfmpegCapabilityProbe(
            settings.ffmpeg_path,
            process_factory=process_factory,
            resolver=resolver,
        )
        self._resolved_executable: str | None = None
        self._validation_lock = asyncio.Lock()

    async def validate(self) -> None:
        if self._resolved_executable is not None:
            return
        async with self._validation_lock:
            if self._resolved_executable is None:
                self._resolved_executable = await self._probe.validate()

    async def open(self, stream_url: str) -> FfmpegMusicDecoder:
        if not stream_url.strip():
            raise ValueError("FFmpeg music stream URL must not be empty")
        await self.validate()
        assert self._resolved_executable is not None
        return await FfmpegMusicDecoder.open(
            self._resolved_executable,
            stream_url,
            self._settings,
            process_factory=self._process_factory,
        )


class FfmpegMusicDecoder:
    """Read one FFmpeg child stdout without blocking or leaking the process."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        stream_url: str,
        settings: AudioMixerSettings,
    ) -> None:
        if process.stdout is None or process.stderr is None:
            raise ValueError("FFmpeg decoder requires stdout and stderr pipes")
        self._process = process
        self._source_ref = opaque_ref(stream_url)
        self._settings = settings
        self._stdout = process.stdout
        self._stderr = process.stderr
        self._stderr_tail = bytearray()
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(),
            name=f"ffmpeg-stderr:{self._source_ref}",
        )
        self._pcm = bytearray()
        self._prefetched: DecodedAudioBlock | None = None
        self._eof = False
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._stats = FfmpegDecoderStats()

    @classmethod
    async def open(
        cls,
        executable: str,
        stream_url: str,
        settings: AudioMixerSettings,
        *,
        process_factory: _ProcessFactory = asyncio.create_subprocess_exec,
    ) -> FfmpegMusicDecoder:
        command = cls.command(executable, stream_url, settings)
        try:
            process = await process_factory(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise MusicDecoderError(
                f"Could not start FFmpeg decoder: {exception_kind(exc)}"
            ) from exc
        decoder = cls(process, stream_url, settings)
        try:
            async with asyncio.timeout(settings.decoder_start_timeout_seconds):
                decoder._prefetched = await decoder._read_block()
            if decoder._prefetched is None:
                raise MusicDecoderError("FFmpeg produced no audio")
        except TimeoutError as exc:
            await asyncio.shield(decoder.aclose())
            raise MusicDecoderError("FFmpeg decoder startup timed out") from exc
        except BaseException:
            await asyncio.shield(decoder.aclose())
            raise
        logger.info("FFmpeg music decoder started: source=%s", decoder._source_ref)
        return decoder

    @staticmethod
    def command(
        executable: str,
        stream_url: str,
        settings: AudioMixerSettings,
    ) -> tuple[str, ...]:
        rw_timeout_us = max(1, round(settings.decoder_read_timeout_seconds * 1_000_000))
        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
        ]
        if urlparse(stream_url).scheme.casefold() in {"http", "https"}:
            command.extend(
                (
                    "-rw_timeout",
                    str(rw_timeout_us),
                    "-reconnect",
                    "1",
                    "-reconnect_on_network_error",
                    "1",
                    "-reconnect_on_http_error",
                    "429,5xx",
                    "-reconnect_streamed",
                    "1",
                    "-reconnect_delay_max",
                    "2",
                )
            )
        command.extend(
            (
                "-i",
                stream_url,
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
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
        )
        return tuple(command)

    @property
    def stats(self) -> FfmpegDecoderStats:
        return self._stats

    def __aiter__(self) -> AsyncIterator[DecodedAudioBlock]:
        return self

    async def __anext__(self) -> DecodedAudioBlock:
        if self._closed:
            raise StopAsyncIteration
        if self._prefetched is not None:
            block = self._prefetched
            self._prefetched = None
            return block
        block = await self._read_block()
        if block is None:
            raise StopAsyncIteration
        return block

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            stdout_task = asyncio.create_task(
                self._discard_stdout(),
                name=f"ffmpeg-stdout-discard:{self._source_ref}",
            )
            wait_task = asyncio.create_task(
                self._process.wait(),
                name=f"ffmpeg-wait:{self._source_ref}",
            )
            try:
                if self._process.returncode is None:
                    self._process.terminate()
                try:
                    await self._await_process_exit(wait_task)
                except TimeoutError:
                    if self._process.returncode is None:
                        self._process.kill()
                        self._stats = replace(
                            self._stats,
                            forced_kills=self._stats.forced_kills + 1,
                        )
                    try:
                        await self._await_process_exit(wait_task)
                    except TimeoutError:
                        logger.warning(
                            "FFmpeg process did not settle after forced stop: source=%s",
                            self._source_ref,
                        )
            except asyncio.CancelledError:
                if self._process.returncode is None:
                    self._process.kill()
                    self._stats = replace(
                        self._stats,
                        forced_kills=self._stats.forced_kills + 1,
                    )
                with suppress(BaseException):
                    await self._await_process_exit(wait_task)
                raise
            finally:
                for task in (stdout_task, wait_task):
                    if not task.done():
                        task.cancel()
                if not self._stderr_task.done():
                    self._stderr_task.cancel()
                await asyncio.gather(
                    stdout_task,
                    wait_task,
                    self._stderr_task,
                    return_exceptions=True,
                )
            logger.info("FFmpeg music decoder closed: source=%s", self._source_ref)

    async def _await_process_exit(self, wait_task: asyncio.Task[int]) -> int:
        async with asyncio.timeout(self._settings.decoder_stop_timeout_seconds):
            return await asyncio.shield(wait_task)

    async def _discard_stdout(self) -> None:
        while await self._stdout.read(64 * 1024):
            pass

    async def _read_block(self) -> DecodedAudioBlock | None:
        while len(self._pcm) < _BLOCK_BYTES and not self._eof:
            try:
                async with asyncio.timeout(self._settings.decoder_read_timeout_seconds):
                    data = await self._stdout.read(_BLOCK_BYTES - len(self._pcm))
            except TimeoutError as exc:
                raise MusicDecoderError("FFmpeg PCM read timed out") from exc
            if data:
                self._pcm.extend(data)
            else:
                self._eof = True

        if len(self._pcm) >= _BLOCK_BYTES:
            payload = bytes(self._pcm[:_BLOCK_BYTES])
            del self._pcm[:_BLOCK_BYTES]
            return self._decode_block(payload, AUDIO_BLOCK_FRAMES)
        await self._check_exit()
        if not self._pcm:
            return None
        frame_width = CANONICAL_AUDIO_FORMAT.frame_width_bytes
        if len(self._pcm) % frame_width:
            raise MusicDecoderError("FFmpeg emitted a non-frame-aligned PCM tail")
        valid_frames = len(self._pcm) // frame_width
        payload = bytes(self._pcm)
        self._pcm.clear()
        samples = np.zeros((AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS), dtype=np.float32)
        samples[:valid_frames] = np.frombuffer(payload, dtype="<f4").reshape(
            valid_frames,
            AUDIO_CHANNELS,
        )
        return self._decoded(samples, valid_frames)

    async def _check_exit(self) -> None:
        try:
            async with asyncio.timeout(self._settings.decoder_stop_timeout_seconds):
                returncode = await self._process.wait()
        except TimeoutError as exc:
            raise MusicDecoderError("FFmpeg stdout closed before process exit") from exc
        await asyncio.gather(self._stderr_task, return_exceptions=True)
        if returncode != 0:
            raise MusicDecoderError(f"FFmpeg decoder exited with code {returncode}")
        logger.info("FFmpeg music decoder reached EOF: source=%s", self._source_ref)

    def _decode_block(self, payload: bytes, valid_frames: int) -> DecodedAudioBlock:
        samples = np.frombuffer(payload, dtype="<f4").reshape(
            AUDIO_BLOCK_FRAMES,
            AUDIO_CHANNELS,
        )
        return self._decoded(samples, valid_frames)

    def _decoded(self, samples: np.ndarray, valid_frames: int) -> DecodedAudioBlock:
        invalid = int(np.count_nonzero(~np.isfinite(samples[:valid_frames])))
        if invalid:
            samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
        self._stats = replace(
            self._stats,
            blocks_emitted=self._stats.blocks_emitted + 1,
            valid_frames_emitted=self._stats.valid_frames_emitted + valid_frames,
            invalid_samples=self._stats.invalid_samples + invalid,
        )
        return DecodedAudioBlock(valid_frames, samples)

    async def _drain_stderr(self) -> None:
        while True:
            chunk = await self._stderr.read(4096)
            if not chunk:
                return
            self._stats = replace(
                self._stats,
                stderr_bytes=self._stats.stderr_bytes + len(chunk),
            )
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > _STDERR_TAIL_BYTES:
                del self._stderr_tail[: len(self._stderr_tail) - _STDERR_TAIL_BYTES]
