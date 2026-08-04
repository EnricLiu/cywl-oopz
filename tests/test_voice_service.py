from __future__ import annotations

import asyncio
import time

import pytest

from cywl_oopz.features.voice.errors import (
    VoiceBackendBusyError,
    VoiceRuntimeUnavailableError,
    VoiceSessionAlreadyActiveError,
    VoiceSessionOwnershipError,
    VoiceSessionStartCancelledError,
    VoiceUserNotInChannelError,
)
from cywl_oopz.features.voice.models import (
    VoiceRuntimeStats,
    VoiceSessionState,
    VoiceStartRequest,
    VoiceStopReason,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.service import VoiceConversationService
from cywl_oopz.features.voice.settings import PersistedVoiceSessionStatus
from cywl_oopz.integrations.voice.fake import (
    FakeVoiceAccessGateway,
    FakeVoiceConfigurationRepository,
    FakeVoiceSessionRepository,
    FakeVoiceSessionRuntime,
    FakeVoiceSessionRuntimeFactory,
)
from cywl_oopz.integrations.voice.unavailable import UnavailableVoiceSessionRuntimeFactory
from cywl_oopz.settings import VoiceSettings


def settings(**overrides: str) -> VoiceSettings:
    return VoiceSettings.from_mapping(
        {
            "CYWL_VOICE_ENABLED": "true",
            "CYWL_VOICE_START_TIMEOUT_SECONDS": "1",
            **overrides,
        }
    )


def request(owner: str = "person") -> VoiceStartRequest:
    return VoiceStartRequest(owner, VoiceTextAddress("area", "text"))


def service():
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"
    runtimes = FakeVoiceSessionRuntimeFactory()
    return (
        VoiceConversationService(
            settings(),
            access,
            runtimes,
            FakeVoiceConfigurationRepository(),
            FakeVoiceSessionRepository(),
        ),
        access,
        runtimes,
    )


class RecordingStatusSink:
    def __init__(self) -> None:
        self.statuses = []
        self.closed = False

    @property
    def owns_message(self) -> bool:
        return True

    def emit(self, status) -> None:
        self.statuses.append(status)

    async def aclose(self) -> None:
        self.closed = True


class MemoryContextSource:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[str] = []

    async def context_text(self, person_id: str) -> str:
        self.calls.append(person_id)
        if self.error is not None:
            raise self.error
        return self.text


@pytest.mark.asyncio
async def test_voice_service_runs_fake_session_then_stops_and_releases_everything() -> None:
    conversations, access, runtimes = service()

    active = await conversations.start(request())

    assert active.active is True
    assert active.state is VoiceSessionState.LISTENING
    assert active.voice_channel is not None
    assert runtimes.runtimes[0].started is True
    assert access.active_lease is not None

    stopped = await conversations.stop("person")

    assert stopped.active is False
    assert stopped.state is VoiceSessionState.CLOSED
    assert runtimes.runtimes[0].stop_requests == [VoiceStopReason.COMMAND]
    assert runtimes.runtimes[0].closed is True
    assert access.active_lease is None
    assert access.release_count == 1
    assert (await conversations.status()).active is False
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_streams_runtime_status_and_terminal_usage() -> None:
    conversations, _, runtimes = service()
    display = RecordingStatusSink()

    active = await conversations.start(request(), display)
    runtimes.runtimes[0].set_state(
        VoiceSessionState.SPEAKING,
        VoiceRuntimeStats(responses_started=3, responses_drained=2, task_control_calls=1),
    )
    runtimes.runtimes[0].usage = {
        "input_tokens": 7,
        "output_tokens": 5,
        "total_tokens": 12,
    }
    await conversations.stop("person")

    states = [status.state for status in display.statuses]
    assert states[:3] == [
        VoiceSessionState.STARTING,
        VoiceSessionState.ACQUIRING_VOICE,
        VoiceSessionState.CONNECTING_PROVIDER,
    ]
    assert VoiceSessionState.LISTENING in states
    assert VoiceSessionState.SPEAKING in states
    terminal = display.statuses[-1]
    assert terminal.active is False
    assert terminal.state is VoiceSessionState.CLOSED
    assert terminal.session_id == active.session_id
    assert terminal.model_display_name == "Fake realtime"
    assert terminal.metrics["voice_responses_drained"] == 2
    assert terminal.usage["total_tokens"] == 12
    assert display.closed is True
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_pins_fresh_configuration_and_persists_lifecycle() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"
    runtimes = FakeVoiceSessionRuntimeFactory()
    configurations = FakeVoiceConfigurationRepository()
    sessions = FakeVoiceSessionRepository()
    conversations = VoiceConversationService(settings(), access, runtimes, configurations, sessions)

    active = await conversations.start(request())

    assert len(configurations.resolve_calls) == 1
    assert sessions.created[0][0].session_id == active.session_id
    assert sessions.active == [active.session_id]
    assert runtimes.contexts[0].configuration is sessions.created[0][1]
    runtimes.runtimes[0].usage = {"input_tokens": 4, "output_tokens": 2}

    await conversations.stop("person")

    assert sessions.finished == [(active.session_id, PersistedVoiceSessionStatus.ENDED, "command")]
    assert sessions.finished_usage == [{"input_tokens": 4, "output_tokens": 2}]
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_loads_bounded_memory_once_for_the_runtime() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"
    runtimes = FakeVoiceSessionRuntimeFactory()
    memory = MemoryContextSource("  用户喜欢电子音乐。  ")
    conversations = VoiceConversationService(
        settings(),
        access,
        runtimes,
        FakeVoiceConfigurationRepository(),
        FakeVoiceSessionRepository(),
        memory,
    )

    await conversations.start(request())

    assert memory.calls == ["person"]
    assert runtimes.contexts[0].memory_context == "用户喜欢电子音乐。"
    await conversations.stop("person")
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_continues_when_memory_projection_fails() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"
    runtimes = FakeVoiceSessionRuntimeFactory()
    conversations = VoiceConversationService(
        settings(),
        access,
        runtimes,
        FakeVoiceConfigurationRepository(),
        FakeVoiceSessionRepository(),
        MemoryContextSource(error=RuntimeError("fixture database failure")),
    )

    active = await conversations.start(request())

    assert active.active is True
    assert runtimes.contexts[0].memory_context == ""
    await conversations.stop("person")
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_rejects_second_session_and_non_owner_stop() -> None:
    conversations, access, _ = service()
    access.channels[("area", "other")] = "voice"
    await conversations.start(request())

    with pytest.raises(VoiceSessionAlreadyActiveError):
        await conversations.start(request("other"))
    with pytest.raises(VoiceSessionOwnershipError):
        await conversations.stop("other")

    await conversations.stop("person")
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_reports_missing_channel_and_busy_backend() -> None:
    access = FakeVoiceAccessGateway()
    conversations = VoiceConversationService(
        settings(),
        access,
        FakeVoiceSessionRuntimeFactory(),
        FakeVoiceConfigurationRepository(),
        FakeVoiceSessionRepository(),
    )

    with pytest.raises(VoiceUserNotInChannelError):
        await conversations.start(request())
    assert (await conversations.status()).active is False

    access.channels[("area", "person")] = "voice"
    access.force_busy = True
    with pytest.raises(VoiceBackendBusyError):
        await conversations.start(request())
    assert (await conversations.status()).active is False
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_releases_lease_when_runtime_is_unavailable() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"
    conversations = VoiceConversationService(
        settings(),
        access,
        UnavailableVoiceSessionRuntimeFactory(),
        FakeVoiceConfigurationRepository(),
        FakeVoiceSessionRepository(),
    )

    with pytest.raises(VoiceRuntimeUnavailableError):
        await conversations.start(request())

    assert access.release_count == 1
    assert (await conversations.status()).active is False
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_cleans_up_after_runtime_ends_without_stop_command() -> None:
    conversations, access, runtimes = service()
    await conversations.start(request())

    await runtimes.runtimes[0].finish()
    for _ in range(10):
        if not (await conversations.status()).active:
            break
        await asyncio.sleep(0)

    assert (await conversations.status()).active is False
    assert access.release_count == 1
    assert runtimes.runtimes[0].closed is True
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_persists_provider_terminal_as_failed_with_usage() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"
    runtimes = FakeVoiceSessionRuntimeFactory()
    sessions = FakeVoiceSessionRepository()
    conversations = VoiceConversationService(
        settings(),
        access,
        runtimes,
        FakeVoiceConfigurationRepository(),
        sessions,
    )
    active = await conversations.start(request())
    runtimes.runtimes[0].usage = {"total_tokens": 7}

    await runtimes.runtimes[0].finish(VoiceStopReason.PROVIDER_FAILED)
    for _ in range(10):
        if not (await conversations.status()).active:
            break
        await asyncio.sleep(0)

    assert sessions.finished == [
        (active.session_id, PersistedVoiceSessionStatus.FAILED, "provider_failed")
    ]
    assert sessions.finished_usage == [{"total_tokens": 7}]
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_shutdown_stops_active_runtime_idempotently() -> None:
    conversations, access, runtimes = service()
    await conversations.start(request())

    await conversations.aclose()
    await conversations.aclose()

    assert runtimes.runtimes[0].stop_requests == [VoiceStopReason.SHUTDOWN]
    assert access.release_count == 1


@pytest.mark.asyncio
async def test_voice_service_stop_during_startup_cancels_generation_and_releases_lease() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"

    class BlockingFactory:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.resume = asyncio.Event()
            self.delegate = FakeVoiceSessionRuntimeFactory()

        async def create(self, context):
            self.entered.set()
            await self.resume.wait()
            return await self.delegate.create(context)

    factory = BlockingFactory()
    conversations = VoiceConversationService(
        settings(),
        access,
        factory,
        FakeVoiceConfigurationRepository(),
        FakeVoiceSessionRepository(),
    )
    starting = asyncio.create_task(conversations.start(request()))
    await factory.entered.wait()

    observed = await conversations.status()
    assert observed.active is True
    assert observed.state is VoiceSessionState.CONNECTING_PROVIDER

    stopping = asyncio.create_task(conversations.stop("person"))
    await asyncio.sleep(0)
    factory.resume.set()

    with pytest.raises(VoiceSessionStartCancelledError):
        await starting
    stopped = await stopping
    assert stopped.active is False
    assert stopped.state is VoiceSessionState.CLOSED
    assert access.release_count == 1
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_forces_bounded_cleanup_when_runtime_ignores_stop() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"

    class StubbornRuntime(FakeVoiceSessionRuntime):
        async def request_stop(self, reason):
            self.stop_requests.append(reason)

    class StubbornFactory:
        def __init__(self) -> None:
            self.runtime = StubbornRuntime()

        async def create(self, context):
            del context
            return self.runtime

    factory = StubbornFactory()
    conversations = VoiceConversationService(
        settings(CYWL_VOICE_START_TIMEOUT_SECONDS="0.01"),
        access,
        factory,
        FakeVoiceConfigurationRepository(),
        FakeVoiceSessionRepository(),
    )
    await conversations.start(request())

    stopped = await conversations.stop("person")

    assert stopped.state is VoiceSessionState.CLOSED
    assert factory.runtime.closed is True
    assert access.release_count == 1
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_stop_budget_forces_unresponsive_runtime_and_releases_lease() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"
    never = asyncio.Event()
    persistence_cancelled = asyncio.Event()

    class UnresponsiveSessionRepository(FakeVoiceSessionRepository):
        async def finish(self, *args, **kwargs) -> None:
            del args, kwargs
            try:
                await never.wait()
            finally:
                persistence_cancelled.set()

    class UnresponsiveStatusSink(RecordingStatusSink):
        def __init__(self) -> None:
            super().__init__()
            self.close_cancelled = asyncio.Event()

        async def aclose(self) -> None:
            try:
                await never.wait()
            finally:
                self.close_cancelled.set()

    class UnresponsiveRuntime(FakeVoiceSessionRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.stop_cancelled = asyncio.Event()
            self.close_cancelled = asyncio.Event()

        async def request_stop(self, reason):
            self.stop_requests.append(reason)
            try:
                await never.wait()
            finally:
                self.stop_cancelled.set()

        async def aclose(self) -> None:
            try:
                await never.wait()
            finally:
                self.closed = True
                self.close_cancelled.set()

    class UnresponsiveFactory:
        def __init__(self) -> None:
            self.runtime = UnresponsiveRuntime()

        async def create(self, context):
            self.runtime._context = context
            return self.runtime

    factory = UnresponsiveFactory()
    status_sink = UnresponsiveStatusSink()
    conversations = VoiceConversationService(
        settings(CYWL_VOICE_STOP_TIMEOUT_SECONDS="1.6"),
        access,
        factory,
        FakeVoiceConfigurationRepository(),
        UnresponsiveSessionRepository(),
    )
    await conversations.start(request(), status_sink)

    started_at = time.monotonic()
    async with asyncio.timeout(2):
        stopped = await conversations.stop("person")
    elapsed = time.monotonic() - started_at

    assert elapsed < 2
    assert stopped.active is False
    assert stopped.state is VoiceSessionState.CLOSED
    assert factory.runtime.stop_cancelled.is_set()
    assert factory.runtime.close_cancelled.is_set()
    assert persistence_cancelled.is_set()
    assert status_sink.close_cancelled.is_set()
    assert access.active_lease is None
    assert access.release_count == 1
    assert (await conversations.status()).active is False
    await conversations.aclose()


@pytest.mark.asyncio
async def test_voice_service_retries_timed_out_lease_release_in_background() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"
    first_release_cancelled = asyncio.Event()

    class RetryableLease:
        def __init__(self) -> None:
            self.released = False
            self.attempts = 0

        async def release(self) -> bool:
            self.attempts += 1
            if self.attempts == 1:
                try:
                    await asyncio.Event().wait()
                finally:
                    first_release_cancelled.set()
            self.released = True
            access.active_lease = None
            access.release_count += 1
            return True

    lease = RetryableLease()

    async def try_acquire(channel, owner_key):
        access.acquisitions.append((channel, owner_key))
        if access.active_lease is not None:
            return None
        access.active_lease = lease
        return lease

    access.try_acquire = try_acquire
    conversations = VoiceConversationService(
        settings(),
        access,
        FakeVoiceSessionRuntimeFactory(),
        FakeVoiceConfigurationRepository(),
        FakeVoiceSessionRepository(),
    )
    await conversations.start(request())

    stopped = await conversations.stop("person")

    assert stopped.active is False
    assert first_release_cancelled.is_set()
    assert lease.released is False
    assert access.active_lease is lease
    async with asyncio.timeout(1):
        while not lease.released:
            await asyncio.sleep(0.01)
    assert lease.attempts == 2
    assert access.active_lease is None
    assert access.release_count == 1
    await conversations.aclose()
    assert not conversations._lease_release_tasks


@pytest.mark.asyncio
async def test_voice_service_close_cancels_pending_lease_release_retry() -> None:
    access = FakeVoiceAccessGateway()
    access.channels[("area", "person")] = "voice"

    class HangingLease:
        def __init__(self) -> None:
            self.released = False
            self.attempts = 0
            self.cancelled = 0

        async def release(self) -> bool:
            self.attempts += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled += 1

    lease = HangingLease()

    async def try_acquire(channel, owner_key):
        access.acquisitions.append((channel, owner_key))
        access.active_lease = lease
        return lease

    access.try_acquire = try_acquire
    conversations = VoiceConversationService(
        settings(),
        access,
        FakeVoiceSessionRuntimeFactory(),
        FakeVoiceConfigurationRepository(),
        FakeVoiceSessionRepository(),
    )
    await conversations.start(request())
    await conversations.stop("person")

    assert lease.cancelled == 1
    assert conversations._lease_release_tasks
    await conversations.aclose()

    assert lease.attempts == 1
    assert not conversations._lease_release_tasks
