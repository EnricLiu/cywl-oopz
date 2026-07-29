from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import httpx
import pytest

from cywl_oopz.features.agent.models import LlmProvider, ProviderProtocol
from cywl_oopz.features.agent.provider_retry import bind_provider_retry_progress
from cywl_oopz.features.agent.registry import (
    ObservableAsyncOpenAI,
    ProviderClientPool,
)
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[ConversationProgressEvent] = []

    async def emit(self, event: ConversationProgressEvent) -> None:
        self.events.append(event)


def provider(api_key: str = "first-key") -> LlmProvider:
    return LlmProvider(
        id=uuid4(),
        alias="provider",
        display_name="Provider",
        protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
        base_url="https://llm.example/v1",
        api_key=api_key,
        user_selectable=True,
        enabled=True,
        config={"timeout_seconds": 5, "headers": {"X-Client": "cywl"}},
    )


@pytest.mark.asyncio
async def test_client_pool_reuses_unchanged_provider_and_replaces_changed_credentials() -> None:
    pool = ProviderClientPool()
    original_provider = provider()

    first = await pool.get(original_provider)
    reused = await pool.get(original_provider)
    replacement = await pool.get(replace(original_provider, api_key="second-key"))

    assert first is reused
    assert replacement is not first
    assert first.is_closed is True
    assert replacement.headers["X-Client"] == "cywl"

    await pool.aclose()
    assert replacement.is_closed is True


@pytest.mark.asyncio
async def test_openai_client_retries_transient_response_with_visible_progress() -> None:
    requests = 0
    sleeps: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                503,
                headers={"retry-after-ms": "1250"},
                json={"error": {"message": "temporary", "type": "server_error"}},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "恢复啦♪"},
                        "finish_reason": "stop",
                    }
                ],
            },
            request=request,
        )

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = ObservableAsyncOpenAI(
        provider_alias="test-provider",
        base_url="https://llm.example/v1",
        api_key="test-key",
        http_client=http_client,
        max_retries=1,
        retry_sleep=record_sleep,
    )
    progress = RecordingProgress()

    with bind_provider_retry_progress(progress):
        response = await client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "你好"}],
        )

    assert response.choices[0].message.content == "恢复啦♪"
    assert requests == 2
    assert sleeps == [1.25]
    assert [item.kind for item in progress.events] == [
        ProgressKind.MODEL_RETRY,
        ProgressKind.THINKING,
    ]
    retry = progress.events[0]
    assert retry.retry_attempt == 1
    assert retry.retry_max_attempts == 1
    assert retry.retry_delay_seconds == 1.25
    assert retry.retry_reason == "上游服务异常（HTTP 503）"

    await http_client.aclose()
