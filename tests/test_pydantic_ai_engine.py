from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

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


class FakeRegistry:
    def __init__(self, model) -> None:
        self.value = model
        self.closed = False

    async def model(self, reference):
        return self.value

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_engine_maps_context_usage_and_output_without_framework_leakage() -> None:
    captured: dict[str, object] = {}

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured["messages"] = messages
        captured["instructions"] = info.instructions
        return ModelResponse(parts=[TextPart("final answer")])

    registry = FakeRegistry(FunctionModel(respond))
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

    result = await engine.run(request)
    await engine.aclose()

    assert result.output == "final answer"
    assert result.stop_reason is AgentStopReason.COMPLETED
    assert result.model_requests == 1
    assert captured["instructions"] == "system"
    assert len(captured["messages"]) == 3
    assert registry.closed is True
