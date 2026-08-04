from __future__ import annotations

import asyncio

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

    await conversations.stop("person")

    assert sessions.finished == [(active.session_id, PersistedVoiceSessionStatus.ENDED, "command")]
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
