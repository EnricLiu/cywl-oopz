from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.features.agent.context import AgentContextBuilder
from cywl_oopz.features.agent.models import (
    AgentIdentity,
    AgentMessage,
    AgentModelRef,
    AgentThread,
    ModelCapability,
    ProviderProtocol,
)
from cywl_oopz.features.agent.skills.models import AgentSkill
from cywl_oopz.features.agent.summarization import (
    PydanticAiThreadSummarizer,
    ThreadSummarizer,
    ThreadSummaryRequest,
    ThreadSummaryService,
)
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.settings import AgentSettings


def settings() -> AgentSettings:
    return AgentSettings.from_mapping(
        {
            "CYWL_AGENT_SUMMARY_TRIGGER_MESSAGES": "6",
            "CYWL_AGENT_SUMMARY_RETAIN_MESSAGES": "2",
            "CYWL_AGENT_SUMMARY_MAX_CHARACTERS": "80",
            "CYWL_AGENT_SUMMARY_TIMEOUT_SECONDS": "1",
        }
    )


def model_ref() -> AgentModelRef:
    return AgentModelRef(
        provider_id=uuid4(),
        model_id=uuid4(),
        provider_alias="provider",
        model_alias="model",
        remote_model_name="model",
        protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
        capabilities=frozenset({ModelCapability.TOOL_CALLING}),
        fallback_model_id=None,
    )


def thread() -> AgentThread:
    return AgentThread(
        id=uuid4(),
        key=ConversationKey("channel", "area", "channel", "person"),
        selected_model_id=None,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        summary="Earlier facts.",
        summary_through_sequence=2,
        summary_version=1,
        version=3,
    )


class FakeMessages:
    def __init__(self, values: tuple[AgentMessage, ...]) -> None:
        self.values = values
        self.load_after_sequence: int | None = None

    async def load(
        self,
        thread_id: UUID,
        *,
        limit: int,
        after_sequence: int = 0,
    ) -> tuple[AgentMessage, ...]:
        del thread_id
        self.load_after_sequence = after_sequence
        return tuple(item for item in self.values if (item.sequence or 0) > after_sequence)[-limit:]

    async def load_after(
        self,
        thread_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> tuple[AgentMessage, ...]:
        del thread_id
        return tuple(item for item in self.values if (item.sequence or 0) > after_sequence)[:limit]

    async def append(self, *args, **kwargs) -> None:
        del args, kwargs

    async def count(self, thread_id: UUID) -> int:
        del thread_id
        return len(self.values)


class FakeMemory:
    def __init__(self) -> None:
        self.person_ids: list[str] = []

    async def context_text(self, person_id: str) -> str:
        self.person_ids.append(person_id)
        return "- 用户喜欢爵士乐"


class FailingMemory:
    async def context_text(self, person_id: str) -> str:
        del person_id
        raise DatabaseError("memory unavailable")


class FakeThreads:
    def __init__(self) -> None:
        self.saved: tuple[UUID, str, int, int] | None = None

    async def save_summary(
        self,
        thread_id: UUID,
        summary: str,
        through_sequence: int,
        *,
        expected_version: int,
    ) -> bool:
        self.saved = (thread_id, summary, through_sequence, expected_version)
        return True


class RecordingSummarizer(ThreadSummarizer):
    def __init__(self) -> None:
        self.requests: list[ThreadSummaryRequest] = []

    async def summarize(self, request: ThreadSummaryRequest) -> str:
        self.requests.append(request)
        return "Merged summary."


class StaticRegistry:
    def __init__(self, model: FunctionModel) -> None:
        self._model = model

    async def model(self, reference: AgentModelRef) -> FunctionModel:
        del reference
        return self._model


@pytest.mark.asyncio
async def test_context_builder_orders_summary_memory_and_recent_messages() -> None:
    messages = FakeMessages(
        (
            AgentMessage("user", "text", {"text": "old"}, sequence=1),
            AgentMessage("assistant", "text", {"text": "old answer"}, sequence=2),
            AgentMessage("user", "text", {"text": "recent"}, sequence=3),
            AgentMessage("assistant", "text", {"text": "recent answer"}, sequence=4),
        )
    )
    memory = FakeMemory()
    builder = AgentContextBuilder(settings(), messages, memory)
    current_thread = thread()
    skill = AgentSkill(
        id=uuid4(),
        name="web-research",
        display_name="网页研究",
        description="搜索并阅读关键来源。",
        instructions="SECRET INSTRUCTIONS",
        version="2",
        revision=3,
        required_tools=frozenset({"search_web"}),
        resources=(),
        metadata={"private": "metadata"},
    )

    context = await builder.build(
        current_thread,
        AgentIdentity("person", current_thread.key),
        available_skills=(skill,),
    )

    assert [item.kind for item in context] == [
        "text",
        "skill_catalog",
        "summary",
        "memory",
        "text",
        "text",
    ]
    assert messages.load_after_sequence == 2
    assert memory.person_ids == ["person"]
    assert context[0].content["text"].startswith(settings().system_prompt)
    assert "## Agent 工作循环" in context[0].content["text"]
    catalog_text = context[1].content["text"]
    assert "web-research" in catalog_text
    assert "搜索并阅读关键来源" in catalog_text
    assert '"version":"2"' in catalog_text
    assert "SECRET INSTRUCTIONS" not in catalog_text
    assert "search_web" not in catalog_text
    assert "metadata" not in catalog_text
    assert "Earlier facts." in context[2].content["text"]

    without_memory = await AgentContextBuilder(
        settings(),
        messages,
        FailingMemory(),
    ).build(current_thread, AgentIdentity("person", current_thread.key))
    assert "memory" not in [item.kind for item in without_memory]


@pytest.mark.asyncio
async def test_summary_service_selects_complete_turns_and_uses_cas() -> None:
    values = tuple(
        AgentMessage(
            "user" if sequence % 2 else "assistant",
            "text",
            {"text": f"message-{sequence}"},
            sequence=sequence,
        )
        for sequence in range(3, 9)
    )
    messages = FakeMessages(values)
    threads = FakeThreads()
    summarizer = RecordingSummarizer()
    service = ThreadSummaryService(settings(), summarizer, threads, messages)
    current_thread = thread()

    saved = await service.maybe_summarize(current_thread, model_ref())

    assert saved is True
    assert [item.sequence for item in summarizer.requests[0].messages] == [3, 4, 5, 6]
    assert summarizer.requests[0].previous_summary == "Earlier facts."
    assert threads.saved == (
        current_thread.id,
        "Merged summary.",
        6,
        current_thread.version,
    )


@pytest.mark.asyncio
async def test_pydantic_summarizer_uses_bounded_no_tool_request() -> None:
    prompts: list[str] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        prompts.append(str(messages))
        return ModelResponse(parts=[TextPart("摘要" * 100)])

    summarizer = PydanticAiThreadSummarizer(
        StaticRegistry(FunctionModel(respond)),
        settings(),
    )
    request = ThreadSummaryRequest(
        model=model_ref(),
        previous_summary="old summary",
        messages=(
            AgentMessage("user", "text", {"text": "new fact"}),
            AgentMessage(
                "tool",
                "tool_result",
                {"tool_name": "status", "result": {"ok": True}},
            ),
        ),
        max_characters=20,
    )

    result = await summarizer.summarize(request)

    assert len(result) == 20
    assert "new fact" in prompts[0]
    assert "tool_result" in prompts[0]
