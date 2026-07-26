from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from pydantic_ai import Agent, AgentRunResultEvent, RunContext, UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider


@dataclass
class ToolDeps:
    values: list[int] = field(default_factory=list)


async def double(ctx: RunContext[ToolDeps], value: int) -> int:
    """Double one integer and remember the validated input."""
    ctx.deps.values.append(value)
    return value * 2


@pytest.mark.asyncio
async def test_openai_compatible_model_runs_a_tool_loop() -> None:
    payloads: list[dict[str, Any]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        assert request.url == "https://llm.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"

        if len(payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-tool",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "agent-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-double",
                                        "type": "function",
                                        "function": {
                                            "name": "double",
                                            "arguments": '{"value": 3}',
                                        },
                                    },
                                    {
                                        "id": "call-double-again",
                                        "type": "function",
                                        "function": {
                                            "name": "double",
                                            "arguments": '{"value": 4}',
                                        },
                                    },
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-final",
                "object": "chat.completion",
                "created": 2,
                "model": "agent-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "结果是 6 和 8。"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "total_tokens": 24,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    provider = OpenAIProvider(
        base_url="https://llm.example/v1",
        api_key="test-key",
        http_client=client,
    )
    # Avoid OpenAI SDK's unrelated subprocess-based platform probe in the
    # restricted test environment.
    provider.client._platform = "Linux"
    model = OpenAIChatModel("agent-model", provider=provider)
    agent = Agent(model, deps_type=ToolDeps, tools=[double])
    deps = ToolDeps()
    try:
        result = await asyncio.wait_for(
            agent.run(
                "把 3 加倍",
                deps=deps,
                usage_limits=UsageLimits(request_limit=2, tool_calls_limit=2),
            ),
            timeout=5,
        )
    finally:
        await client.aclose()

    assert result.output == "结果是 6 和 8。"
    assert sorted(deps.values) == [3, 4]
    assert len(payloads) == 2
    assert payloads[0]["tools"][0]["function"]["name"] == "double"
    assert payloads[1]["messages"][-2:] == [
        {
            "role": "tool",
            "tool_call_id": "call-double",
            "content": "6",
        },
        {
            "role": "tool",
            "tool_call_id": "call-double-again",
            "content": "8",
        },
    ]


@pytest.mark.asyncio
async def test_invalid_tool_arguments_are_returned_to_the_model() -> None:
    payloads: list[dict[str, Any]] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-invalid-tool",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "agent-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-invalid",
                                        "type": "function",
                                        "function": {
                                            "name": "double",
                                            "arguments": '{"value": "three"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                },
            )

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-recovered",
                "object": "chat.completion",
                "created": 2,
                "model": "agent-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "参数无效，未执行工具。"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "total_tokens": 17,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    provider = OpenAIProvider(
        base_url="https://llm.example/v1",
        api_key="test-key",
        http_client=client,
    )
    provider.client._platform = "Linux"
    agent = Agent(
        OpenAIChatModel("agent-model", provider=provider),
        deps_type=ToolDeps,
        tools=[double],
    )
    deps = ToolDeps()
    try:
        result = await agent.run(
            "错误参数",
            deps=deps,
            usage_limits=UsageLimits(request_limit=2, tool_calls_limit=1),
        )
    finally:
        await client.aclose()

    assert result.output == "参数无效，未执行工具。"
    assert deps.values == []
    retry_message = payloads[1]["messages"][-1]
    assert retry_message["role"] == "tool"
    assert retry_message["tool_call_id"] == "call-invalid"
    assert "valid integer" in retry_message["content"]


@pytest.mark.asyncio
async def test_streaming_events_and_final_output_are_preserved() -> None:
    async def stream_response(*_: object):
        yield "你"
        yield "好"

    agent = Agent(FunctionModel(stream_function=stream_response))
    text_deltas: list[str] = []
    final_output: str | None = None

    async with agent.run_stream_events("打招呼") as events:
        async for event in events:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                text_deltas.append(event.part.content)
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                text_deltas.append(event.delta.content_delta)
            elif isinstance(event, AgentRunResultEvent):
                final_output = event.result.output

    assert "".join(text_deltas) == "你好"
    assert final_output == "你好"


@pytest.mark.asyncio
async def test_agent_request_limit_stops_a_tool_loop() -> None:
    agent = Agent(
        TestModel(call_tools=["double"]),
        deps_type=ToolDeps,
        tools=[double],
    )

    with pytest.raises(UsageLimitExceeded, match="request_limit"):
        await agent.run(
            "调用工具",
            deps=ToolDeps(),
            usage_limits=UsageLimits(request_limit=1),
        )


@pytest.mark.asyncio
async def test_cancellation_reaches_the_model_request() -> None:
    started = asyncio.Event()

    async def wait_forever(*_: object):
        started.set()
        await asyncio.Event().wait()

    agent = Agent(FunctionModel(wait_forever))
    task = asyncio.create_task(agent.run("等待"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
