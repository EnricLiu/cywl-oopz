from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


@dataclass(frozen=True, slots=True)
class LiveLlmConfig:
    base_url: str
    api_key: str
    model_name: str


class SignallingTransport(httpx.AsyncBaseTransport):
    """Expose request dispatch so cancellation is deterministic."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.started.set()
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()


@pytest.mark.asyncio
async def test_live_provider_text_tools_streaming_and_usage() -> None:
    if os.getenv("CYWL_RUN_LIVE_LLM_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_LLM_TESTS=1 to run the local live LLM smoke test")

    config = _load_live_config()
    transport = SignallingTransport()
    client = httpx.AsyncClient(timeout=90, transport=transport)
    provider = OpenAIProvider(
        base_url=config.base_url,
        api_key=config.api_key,
        http_client=client,
    )
    # The OpenAI SDK otherwise probes the WSL platform through a subprocess,
    # which is unrelated to provider compatibility and can block restricted CI.
    provider.client._platform = "Linux"
    model = OpenAIChatModel(config.model_name, provider=provider)
    metrics: dict[str, object] = {"model": config.model_name}
    try:
        started_at = time.perf_counter()
        text_result = await Agent(model).run("只回复“CYWL_LIVE_OK”，不要添加其他文字。")
        metrics["text_seconds"] = round(time.perf_counter() - started_at, 3)
        metrics["text_usage"] = {
            "requests": text_result.usage.requests,
            "input_tokens": text_result.usage.input_tokens,
            "output_tokens": text_result.usage.output_tokens,
        }
        assert "CYWL_LIVE_OK" in text_result.output
        assert text_result.usage.requests == 1
        assert text_result.usage.input_tokens > 0
        assert text_result.usage.output_tokens > 0

        calls: list[int] = []

        async def record_number(value: int) -> str:
            """Record one integer for a deterministic compatibility check."""
            calls.append(value)
            return f"recorded:{value}"

        tool_agent = Agent(model, tools=[record_number])
        started_at = time.perf_counter()
        tool_result = await tool_agent.run(
            "必须调用 record_number 两次，参数依次为 17 和 23；"
            "获得两个工具结果后，用一句话确认完成。"
        )
        metrics["tool_loop_seconds"] = round(time.perf_counter() - started_at, 3)
        metrics["tool_usage"] = {
            "requests": tool_result.usage.requests,
            "tool_calls": tool_result.usage.tool_calls,
            "input_tokens": tool_result.usage.input_tokens,
            "output_tokens": tool_result.usage.output_tokens,
        }
        assert sorted(calls) == [17, 23]
        assert tool_result.output.strip()
        assert tool_result.usage.tool_calls == 2
        assert tool_result.usage.requests >= 2

        started_at = time.perf_counter()
        first_delta_at: float | None = None
        streamed_parts: list[str] = []
        async with Agent(model).run_stream("只回复“流式正常”，不要添加其他文字。") as stream:
            async for delta in stream.stream_text(delta=True):
                if first_delta_at is None:
                    first_delta_at = time.perf_counter()
                streamed_parts.append(delta)
        streamed_output = "".join(streamed_parts)
        assert first_delta_at is not None
        metrics["stream_first_delta_seconds"] = round(first_delta_at - started_at, 3)
        metrics["stream_full_seconds"] = round(time.perf_counter() - started_at, 3)
        assert "流式正常" in streamed_output

        transport.started.clear()
        started_at = time.perf_counter()
        cancelled_run = asyncio.create_task(
            Agent(model).run("写一篇结构完整、内容详尽的长篇科幻故事。")
        )
        await asyncio.wait_for(transport.started.wait(), timeout=10)
        cancelled_run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_run
        metrics["cancel_seconds"] = round(time.perf_counter() - started_at, 3)

        started_at = time.perf_counter()
        recovery = await Agent(model).run("只回复“取消恢复正常”。")
        metrics["recovery_seconds"] = round(time.perf_counter() - started_at, 3)
        assert "取消恢复正常" in recovery.output
    finally:
        await client.aclose()
    if os.getenv("CYWL_LIVE_LLM_REPORT") == "1":
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


def _load_live_config() -> LiveLlmConfig:
    """Load explicit environment values, then the developer-only commented block."""
    explicit = LiveLlmConfig(
        base_url=os.getenv("CYWL_LIVE_LLM_BASE_URL", "").strip(),
        api_key=os.getenv("CYWL_LIVE_LLM_API_KEY", "").strip(),
        model_name=os.getenv("CYWL_LIVE_LLM_MODEL", "").strip(),
    )
    if all((explicit.base_url, explicit.api_key, explicit.model_name)):
        return explicit

    comments: dict[str, str] = {}
    for raw_line in Path(".env").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        for label, field_name in (
            ("# endpoint:", "base_url"),
            ("# api-key:", "api_key"),
            ("# model name:", "model_name"),
        ):
            if line.casefold().startswith(label):
                comments[field_name] = line[len(label) :].strip()
    config = LiveLlmConfig(
        base_url=comments.get("base_url", ""),
        api_key=comments.get("api_key", ""),
        model_name=comments.get("model_name", ""),
    )
    if not all((config.base_url, config.api_key, config.model_name)):
        pytest.fail("Live LLM endpoint, API key, and model name are not configured")
    return config
