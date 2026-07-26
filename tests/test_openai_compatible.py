from __future__ import annotations

import json

import httpx
import pytest

from cywl_oopz.core.errors import ProviderError, ProviderResponseError, ProviderTimeoutError
from cywl_oopz.features.chat.models import ChatMessage, ChatRequest, ChatRole
from cywl_oopz.features.chat.openai_compatible import OpenAICompatibleChatProvider
from cywl_oopz.features.chat.streaming import StreamResponseAssembler


def request() -> ChatRequest:
    return ChatRequest(
        model="test-model",
        messages=(ChatMessage(ChatRole.USER, "hello"),),
        user_id="person-1",
        timeout_seconds=2.0,
    )


@pytest.mark.asyncio
async def test_complete_sends_openai_compatible_payload(chat_settings) -> None:
    async def handler(incoming: httpx.Request) -> httpx.Response:
        assert incoming.url == "https://llm.example/v1/chat/completions"
        assert incoming.headers["authorization"] == "Bearer test-key"
        assert json.loads(incoming.content) == {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "user": "person-1",
            "stream": False,
        }
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [{"message": {"content": "hello back"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleChatProvider(chat_settings, client)

    response = await provider.complete(request())

    assert response.content == "hello back"
    assert response.model == "served-model"
    assert response.input_tokens == 3
    assert response.output_tokens == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_aggregates_sse_without_token_messages(chat_settings) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'data: {"model":"served-model","choices":[{"delta":{"content":"hello "}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleChatProvider(chat_settings, client)

    response = await StreamResponseAssembler().assemble(provider.stream(request()), "test-model")

    assert response.content == "hello world"
    assert response.finish_reason == "stop"
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_hides_non_success_response_body(chat_settings) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="secret provider diagnostic")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleChatProvider(chat_settings, client)

    with pytest.raises(ProviderError) as error:
        await provider.complete(request())

    assert "secret provider diagnostic" not in str(error.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_invalid_stream_event_is_rejected(chat_settings) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: not-json\n\n")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleChatProvider(chat_settings, client)

    with pytest.raises(ProviderResponseError):
        await StreamResponseAssembler().assemble(provider.stream(request()), "test-model")

    await client.aclose()


@pytest.mark.asyncio
async def test_http_timeout_is_mapped_to_provider_timeout(chat_settings) -> None:
    async def handler(incoming: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("provider timeout", request=incoming)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleChatProvider(chat_settings, client)

    with pytest.raises(ProviderTimeoutError):
        await provider.complete(request())

    await client.aclose()
