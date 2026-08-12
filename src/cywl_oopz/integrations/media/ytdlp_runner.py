"""Async supervision for isolated one-request yt-dlp worker processes."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import suppress
from typing import Protocol

from cywl_oopz.features.music.errors import (
    MusicExtractionTimeoutError,
    MusicExtractorProcessError,
    MusicExtractorProtocolError,
    MusicSourceUnavailableError,
)
from cywl_oopz.settings import YtDlpMusicSettings

from .ytdlp_models import (
    YtDlpMode,
    YtDlpOperation,
    YtDlpWorkerConfiguration,
    YtDlpWorkerRequest,
    YtDlpWorkerResponse,
)

logger = logging.getLogger(__name__)

_MAX_REQUEST_BYTES = 16 * 1024
_MAX_STDOUT_BYTES = 256 * 1024
_MAX_STDERR_BYTES = 32 * 1024
_READ_CHUNK_BYTES = 8 * 1024


class _Process(Protocol):
    stdin: asyncio.StreamWriter | None
    stdout: asyncio.StreamReader | None
    stderr: asyncio.StreamReader | None
    returncode: int | None
    pid: int

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class YtDlpCapabilityProbe:
    """Validate installed worker components without contacting media sites."""

    def __init__(self, runner: YtDlpProcessRunner) -> None:
        self._runner = runner

    async def validate(self, *, require_javascript: bool) -> dict[str, str]:
        response = await self._runner.probe(require_javascript=require_javascript)
        if not response.ok:
            assert response.error is not None
            raise MusicSourceUnavailableError(
                f"yt-dlp capability check failed: {response.error.code}"
            )
        return response.capabilities


class YtDlpProcessRunner:
    """Run blocking extraction behind cancellable process-group boundaries."""

    def __init__(
        self,
        settings: YtDlpMusicSettings,
        *,
        process_factory=asyncio.create_subprocess_exec,
        python_executable: str = sys.executable,
        worker_module: str = "cywl_oopz.integrations.media.ytdlp_worker",
    ) -> None:
        self._settings = settings
        self._process_factory = process_factory
        self._python_executable = python_executable
        self._worker_module = worker_module
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._active: set[_Process] = set()
        self._closing = False
        self._close_lock = asyncio.Lock()

    @property
    def active_processes(self) -> int:
        return len(self._active)

    def configuration(
        self,
        *,
        cookie_file: str = "",
        require_javascript: bool = False,
    ) -> YtDlpWorkerConfiguration:
        return YtDlpWorkerConfiguration(
            socket_timeout_seconds=self._settings.socket_timeout_seconds,
            max_audio_bitrate_kbps=self._settings.max_audio_bitrate_kbps,
            cache_dir=self._settings.cache_dir,
            js_runtime=self._settings.js_runtime,
            js_runtime_path=self._settings.js_runtime_path,
            cookie_file=cookie_file,
            require_javascript=require_javascript,
        )

    async def run(self, request: YtDlpWorkerRequest) -> YtDlpWorkerResponse:
        """Run one bounded worker request and validate its response envelope."""
        if self._closing:
            raise MusicExtractorProcessError("yt-dlp runner is closing")
        payload = request.to_bytes()
        if len(payload) > _MAX_REQUEST_BYTES:
            raise MusicExtractorProtocolError("yt-dlp worker request is too large")
        timeout_seconds = (
            self._settings.search_timeout_seconds
            if request.operation is YtDlpOperation.SEARCH
            else self._settings.process_timeout_seconds
        )
        process: _Process | None = None
        started_at = asyncio.get_running_loop().time()
        try:
            async with asyncio.timeout(timeout_seconds):
                async with self._semaphore:
                    if self._closing:
                        raise MusicExtractorProcessError("yt-dlp runner is closing")
                    process = await self._start_process()
                    self._active.add(process)
                    response = await self._exchange(process, payload)
        except asyncio.CancelledError:
            if process is not None:
                await asyncio.shield(self._stop_process(process))
            logger.info(
                "yt-dlp worker cancelled: operation=%s elapsed_ms=%.1f",
                request.operation.value,
                (asyncio.get_running_loop().time() - started_at) * 1_000,
            )
            raise
        except TimeoutError as exc:
            if process is not None:
                await asyncio.shield(self._stop_process(process))
            raise MusicExtractionTimeoutError(
                f"yt-dlp {request.operation.value} timed out"
            ) from exc
        except (MusicExtractorProcessError, MusicExtractorProtocolError):
            if process is not None:
                await asyncio.shield(self._stop_process(process))
            raise
        except Exception as exc:
            if process is not None:
                await asyncio.shield(self._stop_process(process))
            raise MusicExtractorProcessError(f"yt-dlp worker failed: {type(exc).__name__}") from exc
        finally:
            if process is not None:
                self._active.discard(process)
        logger.info(
            "yt-dlp worker completed: operation=%s ok=%s items=%s elapsed_ms=%.1f",
            request.operation.value,
            response.ok,
            len(response.items),
            (asyncio.get_running_loop().time() - started_at) * 1_000,
        )
        return response

    async def probe(self, *, require_javascript: bool) -> YtDlpWorkerResponse:
        return await self.run(
            YtDlpWorkerRequest(
                operation=YtDlpOperation.PROBE,
                mode=YtDlpMode.RUNTIME_ONLY,
                profile="youtube_public" if require_javascript else "generic",
                configuration=self.configuration(require_javascript=require_javascript),
            )
        )

    async def aclose(self) -> None:
        """Reject new work and terminate every active Python/JS process group."""
        async with self._close_lock:
            if self._closing and not self._active:
                return
            self._closing = True
            active = tuple(self._active)
            if active:
                await asyncio.gather(
                    *(self._stop_process(process) for process in active),
                    return_exceptions=True,
                )
            logger.info("yt-dlp process runner closed: terminated=%s", len(active))

    async def _start_process(self) -> _Process:
        try:
            return await self._process_factory(
                self._python_executable,
                "-m",
                self._worker_module,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except Exception as exc:
            raise MusicExtractorProcessError(
                f"Could not start yt-dlp worker: {type(exc).__name__}"
            ) from exc

    async def _exchange(self, process: _Process, payload: bytes) -> YtDlpWorkerResponse:
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise MusicExtractorProcessError("yt-dlp worker pipes are unavailable")
        process.stdin.write(payload)
        await process.stdin.drain()
        process.stdin.close()
        with suppress(Exception):
            await process.stdin.wait_closed()
        stdout_task = asyncio.create_task(
            self._read_bounded(process.stdout, _MAX_STDOUT_BYTES),
            name="ytdlp-worker-stdout",
        )
        stderr_task = asyncio.create_task(
            self._read_bounded(process.stderr, _MAX_STDERR_BYTES),
            name="ytdlp-worker-stderr",
        )
        wait_task = asyncio.create_task(process.wait(), name="ytdlp-worker-wait")
        try:
            stdout, stderr, returncode = await asyncio.gather(
                stdout_task,
                stderr_task,
                wait_task,
            )
        finally:
            for task in (stdout_task, stderr_task, wait_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stdout_task, stderr_task, wait_task, return_exceptions=True)
        if returncode != 0:
            logger.warning(
                "yt-dlp worker exited unsuccessfully: exit_code=%s stderr_bytes=%s",
                returncode,
                len(stderr),
            )
            raise MusicExtractorProcessError("yt-dlp worker exited unsuccessfully")
        try:
            return YtDlpWorkerResponse.from_bytes(stdout)
        except ValueError as exc:
            raise MusicExtractorProtocolError("yt-dlp worker returned invalid JSON") from exc

    @staticmethod
    async def _read_bounded(reader: asyncio.StreamReader, maximum: int) -> bytes:
        output = bytearray()
        while chunk := await reader.read(_READ_CHUNK_BYTES):
            output.extend(chunk)
            if len(output) > maximum:
                raise MusicExtractorProtocolError("yt-dlp worker output exceeded its limit")
        return bytes(output)

    async def _stop_process(self, process: _Process) -> None:
        if process.returncode is not None:
            return
        self._signal_process(process, signal.SIGTERM)
        try:
            async with asyncio.timeout(self._settings.stop_timeout_seconds):
                await process.wait()
            return
        except TimeoutError:
            self._signal_process(process, signal.SIGKILL)
        with suppress(BaseException):
            async with asyncio.timeout(self._settings.stop_timeout_seconds):
                await process.wait()

    @staticmethod
    def _signal_process(process: _Process, signal_number: signal.Signals) -> None:
        if process.returncode is not None:
            return
        if os.name == "posix" and process.pid > 1:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal_number)
                return
        if signal_number == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
