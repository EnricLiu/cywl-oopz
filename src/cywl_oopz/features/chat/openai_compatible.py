"""Async OpenAI-compatible `/chat/completions` provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from cywl_oopz.core.errors import ProviderError, ProviderResponseError, ProviderTimeoutError
from cywl_oopz.settings import ChatSettings

from .models import ChatChunk, ChatRequest, ChatResponse


class OpenAICompatibleChatProvider:
    """Minimal HTTP implementation that works with OpenAI-compatible servers."""

    def __init__(self, settings: ChatSettings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._endpoint = f"{settings.base_url}/chat/completions"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Send a non-streaming chat-completion request and validate its shape."""
        try:
            response = await self._client.post(
                self._endpoint,
                headers=self._headers(),
                json=self._payload(request, stream=False),
                timeout=request.timeout_seconds,
            )
            self._raise_for_provider_error(response)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("LLM request timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("LLM request failed") from exc

        return self._parse_complete_response(response, request.model)

    async def _stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        try:
            async with self._client.stream(
                "POST",
                self._endpoint,
                headers=self._headers(),
                json=self._payload(request, stream=True),
                timeout=request.timeout_seconds,
            ) as response:
                self._raise_for_provider_error(response)
                async for line in response.aiter_lines():
                    chunk = self._parse_sse_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("LLM stream timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("LLM stream failed") from exc

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Return an async iterator for SSE data emitted by the provider."""
        return self._stream(request)

    async def aclose(self) -> None:
        """Close an internally created HTTP client exactly once."""
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _payload(request: ChatRequest, *, stream: bool) -> dict[str, Any]:
        return {
            "model": request.model,
            "messages": [message.to_payload() for message in request.messages],
            "user": request.user_id,
            "stream": stream,
        }

    @staticmethod
    def _raise_for_provider_error(response: httpx.Response) -> None:
        if response.is_error:
            raise ProviderError(f"LLM provider returned HTTP {response.status_code}")

    @staticmethod
    def _parse_complete_response(response: httpx.Response, fallback_model: str) -> ChatResponse:
        try:
            data = response.json()
            choices = data["choices"]
            choice = choices[0]
            content = choice["message"]["content"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderResponseError("LLM provider returned an invalid completion") from exc
        if not isinstance(content, str):
            raise ProviderResponseError("LLM completion content is not text")

        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        return ChatResponse(
            content=content,
            model=str(data.get("model") or fallback_model),
            finish_reason=str(choice.get("finish_reason") or ""),
            input_tokens=OpenAICompatibleChatProvider._optional_int(usage.get("prompt_tokens")),
            output_tokens=OpenAICompatibleChatProvider._optional_int(
                usage.get("completion_tokens")
            ),
        )

    @staticmethod
    def _parse_sse_line(line: str) -> ChatChunk | None:
        if not line.startswith("data:"):
            return None
        raw_data = line.removeprefix("data:").strip()
        if not raw_data or raw_data == "[DONE]":
            return None
        try:
            data = json.loads(raw_data)
            choices = data.get("choices", [])
            choice = choices[0] if choices else {}
            delta = choice.get("delta", {}).get("content", "")
            usage = data.get("usage", {})
        except (AttributeError, IndexError, json.JSONDecodeError) as exc:
            raise ProviderResponseError("LLM provider returned an invalid stream event") from exc
        if delta is None:
            delta = ""
        if not isinstance(delta, str):
            raise ProviderResponseError("LLM stream delta is not text")
        return ChatChunk(
            delta=delta,
            model=str(data.get("model") or ""),
            finish_reason=str(choice.get("finish_reason") or ""),
            input_tokens=OpenAICompatibleChatProvider._optional_int(usage.get("prompt_tokens")),
            output_tokens=OpenAICompatibleChatProvider._optional_int(
                usage.get("completion_tokens")
            ),
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
