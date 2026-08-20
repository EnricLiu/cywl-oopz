from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from cywl_oopz.core.errors import AgentInternalError
from cywl_oopz.features.agent.input import AgentUserInput, ImageInputPart, TextInputPart
from cywl_oopz.features.agent.models import (
    AgentIdentity,
    AgentMessage,
    AgentModelRef,
    AgentRunLimits,
    AgentRunRequest,
    AgentStopReason,
    ProviderProtocol,
)
from cywl_oopz.features.agent.pydantic_ai_engine import PydanticAiAgentEngine
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind


class FakeRegistry:
    def __init__(self, model) -> None:
        self.value = model
        self.closed = False

    async def model(self, reference):
        return self.value

    async def aclose(self) -> None:
        self.closed = True


class RecordingProgress:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[ConversationProgressEvent] = []
        self.fail = fail

    async def emit(self, event: ConversationProgressEvent) -> None:
        self.events.append(event)
        if self.fail:
            raise RuntimeError("display unavailable")


def engine_request() -> AgentRunRequest:
    conversation = ConversationKey("private", "", "", "person")
    return AgentRunRequest(
        run_id=uuid4(),
        thread_id=uuid4(),
        identity=AgentIdentity("person", conversation),
        model=AgentModelRef(
            provider_id=uuid4(),
            model_id=uuid4(),
            provider_alias="provider",
            model_alias="model",
            remote_model_name="remote",
            protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
            capabilities=frozenset(),
            fallback_model_id=None,
        ),
        prompt="current",
        context=(),
        enabled_tools=(),
        limits=AgentRunLimits(),
    )


def test_engine_encodes_current_image_input_as_binary_content() -> None:
    request = replace(
        engine_request(),
        prompt="look",
        user_input=AgentUserInput.from_parts(
            [
                TextInputPart("look"),
                ImageInputPart(data=b"png-bytes", media_type="image/png", byte_size=9),
            ]
        ),
    )

    content = PydanticAiAgentEngine._current_input_content(request)

    assert isinstance(content, list)
    assert content[0] == "look"
    assert content[1].data == b"png-bytes"
    assert content[1].media_type == "image/png"


def test_engine_rehydrates_historical_multimodal_message() -> None:
    instructions, messages = PydanticAiAgentEngine._map_context(
        (
            AgentMessage(
                "user",
                "multimodal",
                {
                    "text": "describe this",
                    "images": [
                        {
                            "data": b"png-bytes",
                            "media_type": "image/png",
                        }
                    ],
                },
            ),
        )
    )

    assert instructions is None
    assert len(messages) == 1
    content = messages[0].parts[0].content
    assert isinstance(content, list)
    assert content[0] == "describe this"
    assert content[1].data == b"png-bytes"


@pytest.mark.asyncio
async def test_openai_chat_wire_maps_binary_image_to_data_url() -> None:
    model = OpenAIChatModel("doubao-seed-2.1-turbo", provider=OpenAIProvider(api_key="test"))

    mapped = await model._map_user_prompt(  # noqa: SLF001 - verify our framework boundary.
        UserPromptPart(
            content=[
                "这是谁",
                BinaryContent(data=b"webp-bytes", media_type="image/webp"),
            ]
        )
    )

    assert mapped["content"] == [
        {"type": "text", "text": "这是谁"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/webp;base64,d2VicC1ieXRlcw=="},
        },
    ]


@pytest.mark.asyncio
async def test_engine_maps_context_usage_and_output_without_framework_leakage() -> None:
    captured: dict[str, object] = {}

    async def respond(messages: list[ModelMessage], info: AgentInfo):
        captured["messages"] = messages
        captured["instructions"] = info.instructions
        yield "final "
        yield "answer"

    registry = FakeRegistry(FunctionModel(stream_function=respond))
    engine = PydanticAiAgentEngine(registry)
    conversation = ConversationKey("private", "", "", "person")
    request = AgentRunRequest(
        run_id=uuid4(),
        thread_id=uuid4(),
        identity=AgentIdentity("person", conversation),
        model=AgentModelRef(
            provider_id=uuid4(),
            model_id=uuid4(),
            provider_alias="provider",
            model_alias="model",
            remote_model_name="remote",
            protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
            capabilities=frozenset(),
            fallback_model_id=None,
        ),
        prompt="current",
        context=(
            AgentMessage("system", "text", {"text": "system"}),
            AgentMessage("user", "text", {"text": "previous"}),
            AgentMessage("assistant", "text", {"text": "earlier answer"}),
        ),
        enabled_tools=(),
        limits=AgentRunLimits(),
    )

    progress = RecordingProgress()
    result = await engine.run(request, progress)
    await engine.aclose()

    assert result.output == "final answer"
    assert result.stop_reason is AgentStopReason.COMPLETED
    assert result.model_requests == 1
    assert captured["instructions"] == "system"
    assert len(captured["messages"]) == 3
    assert registry.closed is True
    assert [event.kind for event in progress.events] == [
        ProgressKind.THINKING,
        ProgressKind.TEXT_RESET,
        ProgressKind.TEXT_DELTA,
        ProgressKind.TEXT_DELTA,
    ]
    assert (
        "".join(event.text for event in progress.events if event.kind is ProgressKind.TEXT_DELTA)
        == "final answer"
    )


@pytest.mark.asyncio
async def test_progress_sink_failure_never_fails_the_agent_run() -> None:
    async def respond(*_: object):
        yield "still works"

    engine = PydanticAiAgentEngine(FakeRegistry(FunctionModel(stream_function=respond)))
    conversation = ConversationKey("private", "", "", "person")
    request = AgentRunRequest(
        run_id=uuid4(),
        thread_id=uuid4(),
        identity=AgentIdentity("person", conversation),
        model=AgentModelRef(
            provider_id=uuid4(),
            model_id=uuid4(),
            provider_alias="provider",
            model_alias="model",
            remote_model_name="remote",
            protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
            capabilities=frozenset(),
            fallback_model_id=None,
        ),
        prompt="current",
        context=(),
        enabled_tools=(),
        limits=AgentRunLimits(),
    )

    result = await engine.run(request, RecordingProgress(fail=True))

    assert result.output == "still works"


@pytest.mark.asyncio
async def test_engine_skips_malformed_history_tool_message() -> None:
    captured: dict[str, object] = {}

    async def respond(messages: list[ModelMessage], _: AgentInfo):
        captured["messages"] = messages
        yield "history recovered"

    engine = PydanticAiAgentEngine(FakeRegistry(FunctionModel(stream_function=respond)))
    request = replace(
        engine_request(),
        context=(
            AgentMessage(
                "assistant",
                "tool_call",
                {
                    "tool_name": 42,
                    "tool_call_id": None,
                    "arguments": ["not", "an", "object"],
                },
            ),
            AgentMessage("user", "text", {"text": "continue"}),
        ),
    )

    result = await engine.run(request)

    assert result.output == "history recovered"
    assert len(captured["messages"]) == 1  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_engine_classifies_bootstrap_failure_as_internal() -> None:
    class FailingRegistry(FakeRegistry):
        async def model(self, reference):
            del reference
            raise RuntimeError("registry invariant failed")

    engine = PydanticAiAgentEngine(FailingRegistry(None))

    with pytest.raises(AgentInternalError, match="bootstrap"):
        await engine.run(engine_request())


@pytest.mark.asyncio
async def test_engine_classifies_unknown_stream_failure_as_internal() -> None:
    async def fail_stream(*_: object):
        raise RuntimeError("framework stream failed")
        yield "unreachable"

    engine = PydanticAiAgentEngine(FakeRegistry(FunctionModel(stream_function=fail_stream)))

    with pytest.raises(AgentInternalError, match="stream"):
        await engine.run(engine_request())


@pytest.mark.asyncio
async def test_engine_classifies_result_mapping_failure_as_internal(monkeypatch) -> None:
    async def respond(*_: object):
        yield "answer"

    def fail_mapping(*_: object):
        raise ValueError("history projection failed")

    monkeypatch.setattr(PydanticAiAgentEngine, "_map_new_tool_messages", fail_mapping)
    engine = PydanticAiAgentEngine(FakeRegistry(FunctionModel(stream_function=respond)))

    with pytest.raises(AgentInternalError, match="result mapping"):
        await engine.run(engine_request())
