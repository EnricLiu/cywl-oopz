from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from cywl_oopz.core.errors import AuthorizationError, ProviderError, RateLimitExceeded
from cywl_oopz.features.chat.models import (
    ChatChunk,
    ChatMessage,
    ChatRole,
    ConversationKey,
    ConversationSession,
)
from cywl_oopz.features.chat.rate_limit import RateLimitService
from cywl_oopz.features.chat.service import ChatService
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor
from cywl_oopz.testing.chat import InMemoryConversationRepository, RecordingChatProvider


def key(person_id: str = "person-1", channel_id: str = "channel-1") -> ConversationKey:
    return ConversationKey("channel", "area-1", channel_id, person_id)


def service(chat_settings, provider=None, repository=None) -> ChatService:
    return ChatService(
        chat_settings,
        provider or RecordingChatProvider(),
        repository or InMemoryConversationRepository(),
    )


@pytest.mark.asyncio
async def test_ask_persists_user_and_assistant_messages(chat_settings) -> None:
    repository = InMemoryConversationRepository()
    provider = RecordingChatProvider(["answer"])

    response = await service(chat_settings, provider, repository).ask(key(), "question")

    assert response.content == "answer"
    session = repository.sessions[key()]
    assert session.messages == (
        ChatMessage(ChatRole.USER, "question"),
        ChatMessage(ChatRole.ASSISTANT, "answer"),
    )
    assert provider.requests[0].messages[0] == ChatMessage(
        ChatRole.SYSTEM, "You are a test assistant."
    )


class OrderedProvider:
    def __init__(self) -> None:
        self.requests = []
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def complete(self, request):
        raise AssertionError("Streaming is expected")

    async def _stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            await self.release_first.wait()
        yield ChatChunk(delta=f"answer-{len(self.requests)}", model=request.model)

    def stream(self, request):
        return self._stream(request)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_same_conversation_requests_are_serialised(chat_settings) -> None:
    provider = OrderedProvider()
    chat_service = service(chat_settings, provider)

    first = asyncio.create_task(chat_service.ask(key(), "first"))
    await asyncio.wait_for(provider.first_started.wait(), timeout=1)
    second = asyncio.create_task(chat_service.ask(key(), "second"))
    await asyncio.sleep(0)
    assert len(provider.requests) == 1
    provider.release_first.set()
    await asyncio.gather(first, second)

    assert [message.content for message in provider.requests[1].messages] == [
        "You are a test assistant.",
        "first",
        "answer-1",
        "second",
    ]


class ParallelProvider:
    def __init__(self) -> None:
        self.entered = 0
        self.both_entered = asyncio.Event()

    async def complete(self, request):
        raise AssertionError("Streaming is expected")

    async def _stream(self, request):
        self.entered += 1
        if self.entered == 2:
            self.both_entered.set()
        await self.both_entered.wait()
        yield ChatChunk(delta="answer", model=request.model)

    def stream(self, request):
        return self._stream(request)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_different_conversations_can_run_concurrently(chat_settings) -> None:
    provider = ParallelProvider()
    chat_service = service(chat_settings, provider)

    responses = await asyncio.wait_for(
        asyncio.gather(
            chat_service.ask(key(channel_id="channel-a"), "one"),
            chat_service.ask(key(person_id="person-2", channel_id="channel-b"), "two"),
        ),
        timeout=1,
    )

    assert provider.entered == 2
    assert [response.content for response in responses] == ["answer", "answer"]


@pytest.mark.asyncio
async def test_expired_session_is_not_sent_to_provider(chat_settings) -> None:
    repository = InMemoryConversationRepository()
    repository.sessions[key()] = ConversationSession(
        key=key(),
        messages=(ChatMessage(ChatRole.USER, "expired"),),
        selected_model=None,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    provider = RecordingChatProvider(["new answer"])

    await service(chat_settings, provider, repository).ask(key(), "fresh")

    assert [message.content for message in provider.requests[0].messages] == [
        "You are a test assistant.",
        "fresh",
    ]


class FailingProvider:
    async def complete(self, request):
        raise ProviderError("no provider")

    async def _stream(self, request):
        raise ProviderError("no provider")
        yield ChatChunk()

    def stream(self, request):
        return self._stream(request)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_provider_failure_does_not_persist_half_a_turn(chat_settings) -> None:
    repository = InMemoryConversationRepository()

    with pytest.raises(ProviderError):
        await service(chat_settings, FailingProvider(), repository).ask(key(), "question")

    assert repository.sessions == {}


@pytest.mark.asyncio
async def test_model_selection_requires_allow_list_and_authorisation(chat_settings) -> None:
    chat_service = service(chat_settings)

    with pytest.raises(AuthorizationError):
        await chat_service.select_model(key(), "alternate-model")

    selected = await chat_service.select_model(key(person_id="model-admin"), "alternate-model")
    assert selected == "alternate-model"

    with pytest.raises(ValueError, match="not allowed"):
        await chat_service.select_model(key(person_id="model-admin"), "unlisted-model")


@pytest.mark.asyncio
async def test_rate_limit_applies_user_cooldown(chat_settings) -> None:
    limiter = RateLimitService(replace(chat_settings, user_cooldown_seconds=10.0))
    lease = await limiter.acquire(key())
    await lease.release()

    with pytest.raises(RateLimitExceeded) as error:
        await limiter.acquire(key())

    assert error.value.scope == "user cooldown"


class CancellableProvider:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request):
        raise AssertionError("Streaming is expected")

    async def _stream(self, request):
        self.started.set()
        await self.release.wait()
        yield ChatChunk(delta="answer after cancellation", model=request.model)

    def stream(self, request):
        return self._stream(request)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_cancellation_releases_session_lock_and_rate_limit(chat_settings) -> None:
    repository = InMemoryConversationRepository()
    provider = CancellableProvider()
    chat_service = service(chat_settings, provider, repository)
    supervisor = ChatTaskSupervisor()

    assert supervisor.start(key(), chat_service.ask(key(), "cancel me")) is True
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    assert await supervisor.cancel(key()) is True
    assert repository.sessions == {}

    provider.release.set()
    response = await asyncio.wait_for(chat_service.ask(key(), "try again"), timeout=1)
    assert response.content == "answer after cancellation"
