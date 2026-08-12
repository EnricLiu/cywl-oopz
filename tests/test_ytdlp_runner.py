from __future__ import annotations

import asyncio
import signal
from collections import deque

import pytest

import cywl_oopz.integrations.media.ytdlp_runner as runner_module
from cywl_oopz.features.music.errors import (
    MusicExtractionTimeoutError,
    MusicExtractorProcessError,
    MusicExtractorProtocolError,
)
from cywl_oopz.integrations.media.ytdlp_models import (
    YtDlpMode,
    YtDlpOperation,
    YtDlpWorkerRequest,
    YtDlpWorkerResponse,
)
from cywl_oopz.integrations.media.ytdlp_runner import (
    YtDlpCapabilityProbe,
    YtDlpProcessRunner,
)
from cywl_oopz.settings import YtDlpMusicSettings


def settings(**changes: str) -> YtDlpMusicSettings:
    return YtDlpMusicSettings.from_mapping(
        {
            "CYWL_MUSIC_YTDLP_SEARCH_TIMEOUT_SECONDS": "0.1",
            "CYWL_MUSIC_YTDLP_PROCESS_TIMEOUT_SECONDS": "0.2",
            "CYWL_MUSIC_YTDLP_STOP_TIMEOUT_SECONDS": "0.02",
            "CYWL_MUSIC_YTDLP_CACHE_DIR": "/tmp/cywl-ytdlp-runner-test",
            **changes,
        }
    )


def request(operation: YtDlpOperation = YtDlpOperation.LOOKUP) -> YtDlpWorkerRequest:
    return YtDlpWorkerRequest(
        operation=operation,
        mode=(
            YtDlpMode.FLAT_SEARCH if operation is YtDlpOperation.SEARCH else YtDlpMode.FULL_METADATA
        ),
        target="fixture",
    )


class FakeStdin:
    def __init__(self) -> None:
        self.data = bytearray()
        self.closed = False

    def write(self, value: bytes) -> None:
        self.data.extend(value)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class FakeProcess:
    def __init__(
        self,
        stdout: bytes | None,
        *,
        stderr: bytes = b"",
        returncode: int = 0,
    ) -> None:
        self.stdin = FakeStdin()
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode: int | None = None if stdout is None else returncode
        self.pid = 1
        self.terminate_calls = 0
        self.kill_calls = 0
        self._done = asyncio.Event()
        if stdout is not None:
            self.stdout.feed_data(stdout)
            self.stdout.feed_eof()
            self.stderr.feed_data(stderr)
            self.stderr.feed_eof()
            self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    def finish(self, stdout: bytes, *, returncode: int = 0, stderr: bytes = b"") -> None:
        if self._done.is_set():
            return
        self.returncode = returncode
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self._done.set()

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.finish(b"", returncode=-15)

    def kill(self) -> None:
        self.kill_calls += 1
        self.finish(b"", returncode=-9)


class FakeProcessFactory:
    def __init__(self, processes: tuple[FakeProcess, ...]) -> None:
        self._processes = deque(processes)
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def __call__(self, *args, **kwargs) -> FakeProcess:
        self.calls.append((args, kwargs))
        return self._processes.popleft()


@pytest.mark.asyncio
async def test_ytdlp_runner_round_trips_one_bounded_json_request() -> None:
    expected = YtDlpWorkerResponse(ok=True, capabilities={"yt_dlp": "fixture"})
    process = FakeProcess(expected.to_bytes())
    factory = FakeProcessFactory((process,))
    runner = YtDlpProcessRunner(settings(), process_factory=factory)

    response = await runner.run(request())

    assert response == expected
    assert YtDlpWorkerRequest.from_bytes(bytes(process.stdin.data)) == request()
    assert process.stdin.closed is True
    assert factory.calls[0][0][1:3] == ("-m", "cywl_oopz.integrations.media.ytdlp_worker")
    assert factory.calls[0][1]["start_new_session"] is True
    assert runner.active_processes == 0
    await runner.aclose()


@pytest.mark.asyncio
async def test_ytdlp_runner_rejects_invalid_and_oversized_stdout() -> None:
    invalid = FakeProcess(b"not-json")
    oversized = FakeProcess(b"x" * (256 * 1024 + 1))
    runner = YtDlpProcessRunner(
        settings(),
        process_factory=FakeProcessFactory((invalid, oversized)),
    )

    with pytest.raises(MusicExtractorProtocolError, match="invalid JSON"):
        await runner.run(request())
    with pytest.raises(MusicExtractorProtocolError, match="exceeded"):
        await runner.run(request())

    await runner.aclose()


@pytest.mark.asyncio
async def test_ytdlp_runner_timeout_and_cancellation_stop_the_worker() -> None:
    timed_out = FakeProcess(None)
    cancelled = FakeProcess(None)
    runner = YtDlpProcessRunner(
        settings(CYWL_MUSIC_YTDLP_SEARCH_TIMEOUT_SECONDS="0.01"),
        process_factory=FakeProcessFactory((timed_out, cancelled)),
    )

    with pytest.raises(MusicExtractionTimeoutError):
        await runner.run(request(YtDlpOperation.SEARCH))
    assert timed_out.terminate_calls == 1

    task = asyncio.create_task(runner.run(request()))
    while runner.active_processes == 0:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.terminate_calls == 1
    assert runner.active_processes == 0
    await runner.aclose()


@pytest.mark.asyncio
async def test_ytdlp_runner_serializes_work_at_the_configured_limit() -> None:
    first = FakeProcess(None)
    second = FakeProcess(YtDlpWorkerResponse(ok=True).to_bytes())
    factory = FakeProcessFactory((first, second))
    runner = YtDlpProcessRunner(
        settings(
            CYWL_MUSIC_YTDLP_MAX_CONCURRENCY="1",
            CYWL_MUSIC_YTDLP_PROCESS_TIMEOUT_SECONDS="1",
        ),
        process_factory=factory,
    )

    first_task = asyncio.create_task(runner.run(request()))
    while len(factory.calls) < 1:
        await asyncio.sleep(0)
    second_task = asyncio.create_task(runner.run(request()))
    await asyncio.sleep(0.01)
    assert len(factory.calls) == 1

    first.finish(YtDlpWorkerResponse(ok=True).to_bytes())
    await first_task
    await second_task
    assert len(factory.calls) == 2
    await runner.aclose()


@pytest.mark.asyncio
async def test_ytdlp_runner_close_terminates_active_work() -> None:
    process = FakeProcess(None)
    runner = YtDlpProcessRunner(
        settings(CYWL_MUSIC_YTDLP_PROCESS_TIMEOUT_SECONDS="1"),
        process_factory=FakeProcessFactory((process,)),
    )
    task = asyncio.create_task(runner.run(request()))
    while runner.active_processes == 0:
        await asyncio.sleep(0)

    await runner.aclose()

    with pytest.raises(MusicExtractorProcessError):
        await task
    assert process.terminate_calls == 1
    assert runner.active_processes == 0


@pytest.mark.asyncio
async def test_ytdlp_runner_signals_the_whole_posix_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(None)
    process.pid = 42
    signals: list[tuple[int, signal.Signals]] = []

    def kill_group(pid: int, sent: signal.Signals) -> None:
        signals.append((pid, sent))
        process.finish(b"", returncode=-int(sent))

    monkeypatch.setattr(runner_module.os, "killpg", kill_group)
    runner = YtDlpProcessRunner(
        settings(CYWL_MUSIC_YTDLP_PROCESS_TIMEOUT_SECONDS="1"),
        process_factory=FakeProcessFactory((process,)),
    )
    task = asyncio.create_task(runner.run(request()))
    while runner.active_processes == 0:
        await asyncio.sleep(0)

    await runner.aclose()

    with pytest.raises(MusicExtractorProcessError):
        await task
    assert signals == [(42, signal.SIGTERM)]
    assert process.terminate_calls == 0


@pytest.mark.asyncio
async def test_real_ytdlp_worker_probe_reports_locked_packages_without_network() -> None:
    runner = YtDlpProcessRunner(settings(CYWL_MUSIC_YTDLP_PROCESS_TIMEOUT_SECONDS="5"))
    capabilities = await YtDlpCapabilityProbe(runner).validate(require_javascript=False)

    assert capabilities["yt_dlp"]
    assert capabilities["yt_dlp_ejs"]
    await runner.aclose()
