from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from cywl_oopz.features.agent.models import (
    AgentIdentity,
    AgentModelRef,
    AgentRunLimits,
    AgentRunRequest,
    AgentStopReason,
    ModelCapability,
    ProviderProtocol,
)
from cywl_oopz.features.agent.pydantic_ai_engine import PydanticAiAgentEngine
from cywl_oopz.features.agent.tools.models import (
    ToolCall,
    ToolDescriptor,
    ToolEffect,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from cywl_oopz.features.agent.tools.web import SearchWebInput, SearchWebOutput
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind


class ValueInput(BaseModel):
    value: int


class ValueOutput(BaseModel):
    value: int


class StaticRegistry:
    def __init__(self, model: FunctionModel) -> None:
        self._model = model

    async def model(self, _):
        return self._model

    async def aclose(self) -> None:
        return None


class RecordingRuntime:
    def __init__(self, names: tuple[str, ...] = ("double",)) -> None:
        self._descriptors = {
            name: ToolDescriptor(
                name=name,
                display_name=name,
                description=f"Execute {name}.",
                input_model=ValueInput,
                output_model=ValueOutput,
                effect=ToolEffect.READ,
                timeout_seconds=1,
                max_output_characters=1000,
                concurrency_safe=True,
                idempotent=True,
            )
            for name in names
        }
        self.calls: list[tuple[ToolCall, ToolExecutionContext]] = []
        self.active = 0
        self.max_active = 0

    def descriptors(self, names: tuple[str, ...]) -> tuple[ToolDescriptor, ...]:
        return tuple(self._descriptors[name] for name in sorted(names))

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        self.calls.append((call, context))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ToolExecutionResult(
            call.call_id,
            call.name,
            ToolExecutionStatus.SUCCEEDED,
            {"value": int(call.arguments["value"]) * 2},
        )


class FailingRuntime(RecordingRuntime):
    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        self.calls.append((call, context))
        return ToolExecutionResult(
            call.call_id,
            call.name,
            ToolExecutionStatus.FAILED,
            error_code="tool_failed",
        )


class SearchWebRuntime:
    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name="search_web",
            display_name="搜索网页",
            description="Search the public web.",
            input_model=SearchWebInput,
            output_model=SearchWebOutput,
            effect=ToolEffect.READ,
            timeout_seconds=1,
            max_output_characters=4000,
            concurrency_safe=True,
            idempotent=True,
        )
        self.calls: list[ToolCall] = []

    def descriptors(self, names: tuple[str, ...]) -> tuple[ToolDescriptor, ...]:
        return (self._descriptor,) if "search_web" in names else ()

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        del context
        self.calls.append(call)
        return ToolExecutionResult(
            call.call_id,
            call.name,
            ToolExecutionStatus.SUCCEEDED,
            {
                "query": str(call.arguments["query"]),
                "results": [
                    {
                        "title": "Example source",
                        "url": "https://example.com/current",
                        "snippet": "Current information.",
                    }
                ],
            },
        )


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[ConversationProgressEvent] = []

    async def emit(self, event: ConversationProgressEvent) -> None:
        self.events.append(event)


def streaming_model(respond) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo):
        response = await respond(messages, info)
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {
                    index: DeltaToolCall(
                        name=part.tool_name,
                        json_args=json.dumps(part.args),
                        tool_call_id=part.tool_call_id,
                    )
                }

    return FunctionModel(stream_function=stream)


def request(
    *,
    enabled_tools: tuple[str, ...],
    limits: AgentRunLimits | None = None,
) -> AgentRunRequest:
    key = ConversationKey("channel", "area", "channel", "person")
    return AgentRunRequest(
        run_id=uuid4(),
        thread_id=uuid4(),
        identity=AgentIdentity(
            "person",
            key,
            source_message_id="message",
            transport_channel_id="channel",
        ),
        model=AgentModelRef(
            provider_id=uuid4(),
            model_id=uuid4(),
            provider_alias="provider",
            model_alias="model",
            remote_model_name="model",
            protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
            capabilities=frozenset({ModelCapability.TOOL_CALLING}),
            fallback_model_id=None,
        ),
        prompt="use tools",
        context=(),
        enabled_tools=enabled_tools,
        limits=limits or AgentRunLimits(),
    )


def has_tool_returns(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(message, ModelRequest)
        and any(isinstance(part, ToolReturnPart) for part in message.parts)
        for message in messages
    )


@pytest.mark.asyncio
async def test_engine_runs_tool_loop_and_maps_provider_neutral_pairs() -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if has_tool_returns(messages):
            return ModelResponse(parts=[TextPart("工具执行完成。")])
        return ModelResponse(parts=[ToolCallPart("double", {"value": 4}, "call-double")])

    runtime = RecordingRuntime()
    engine = PydanticAiAgentEngine(
        StaticRegistry(streaming_model(respond)),
        runtime,
    )
    progress = RecordingProgress()
    result = await engine.run(request(enabled_tools=("double",)), progress)

    assert result.output == "工具执行完成。"
    assert result.stop_reason is AgentStopReason.COMPLETED
    assert result.tool_calls == 1
    assert [item.kind for item in result.intermediate_messages] == [
        "tool_call",
        "tool_result",
    ]
    assert runtime.calls[0][0].call_id == "call-double"
    assert runtime.calls[0][1].identity.person_id == "person"
    assert [event.kind for event in progress.events] == [
        ProgressKind.THINKING,
        ProgressKind.TOOL_STARTED,
        ProgressKind.TOOL_SUCCEEDED,
        ProgressKind.TEXT_RESET,
        ProgressKind.TEXT_DELTA,
        ProgressKind.COMPLETED,
    ]
    assert progress.events[1].tool_display_name == "double"


@pytest.mark.asyncio
async def test_engine_returns_tool_failure_as_data_and_allows_model_recovery() -> None:
    observed_result: object = None

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal observed_result
        del info
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, ToolReturnPart):
                        observed_result = part.content
                        return ModelResponse(parts=[TextPart("工具失败，但对话仍可继续。")])
        return ModelResponse(parts=[ToolCallPart("double", {"value": 4}, "call-fail")])

    engine = PydanticAiAgentEngine(
        StaticRegistry(streaming_model(respond)),
        FailingRuntime(),
    )

    result = await engine.run(request(enabled_tools=("double",)))

    assert result.stop_reason is AgentStopReason.COMPLETED
    assert observed_result == {"ok": False, "error": "tool_failed"}


@pytest.mark.asyncio
async def test_engine_can_search_then_return_a_cited_answer() -> None:
    observed_result: object = None

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal observed_result
        del info
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, ToolReturnPart):
                        observed_result = part.content
                        return ModelResponse(
                            parts=[TextPart("查到当前资料，来源：https://example.com/current")]
                        )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "search_web",
                    {"query": "current topic", "time_range": "w"},
                    "call-search",
                )
            ]
        )

    runtime = SearchWebRuntime()
    engine = PydanticAiAgentEngine(
        StaticRegistry(streaming_model(respond)),
        runtime,
    )

    result = await engine.run(request(enabled_tools=("search_web",)))

    assert result.stop_reason is AgentStopReason.COMPLETED
    assert "https://example.com/current" in result.output
    assert runtime.calls[0].arguments == {
        "query": "current topic",
        "time_range": "w",
    }
    assert observed_result == {
        "ok": True,
        "data": {
            "query": "current topic",
            "results": [
                {
                    "title": "Example source",
                    "url": "https://example.com/current",
                    "snippet": "Current information.",
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_engine_enforces_tool_call_and_parallel_budgets() -> None:
    names = ("tool_a", "tool_b", "tool_c")

    async def parallel_response(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        del info
        if has_tool_returns(messages):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[
                ToolCallPart(name, {"value": index}, f"call-{index}")
                for index, name in enumerate(names, start=1)
            ]
        )

    runtime = RecordingRuntime(names)
    engine = PydanticAiAgentEngine(
        StaticRegistry(streaming_model(parallel_response)),
        runtime,
    )
    result = await engine.run(
        request(
            enabled_tools=names,
            limits=AgentRunLimits(
                max_tool_calls=3,
                max_parallel_tools=2,
            ),
        )
    )

    assert result.stop_reason is AgentStopReason.COMPLETED
    assert result.tool_calls == 3
    assert runtime.max_active == 2

    async def endless_tools(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        del messages, info
        return ModelResponse(parts=[ToolCallPart("double", {"value": 1}, f"call-{uuid4()}")])

    limited = PydanticAiAgentEngine(
        StaticRegistry(streaming_model(endless_tools)),
        RecordingRuntime(),
    )
    exhausted = await limited.run(
        request(
            enabled_tools=("double",),
            limits=AgentRunLimits(
                max_model_requests=3,
                max_tool_calls=1,
            ),
        )
    )

    assert exhausted.stop_reason is AgentStopReason.TOOL_CALL_LIMIT
    assert "预算" in exhausted.output
