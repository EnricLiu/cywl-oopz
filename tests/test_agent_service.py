from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from cywl_oopz.core.errors import ProviderError
from cywl_oopz.features.agent.catalog import ReloadableProviderCatalog
from cywl_oopz.features.agent.models import (
    AgentMessage,
    AgentRun,
    AgentRunResult,
    AgentRunState,
    AgentRunStatus,
    AgentStopReason,
    AgentThread,
    LlmModel,
    LlmProvider,
    ModelCapability,
    ModelSelectionCandidates,
    ProviderProtocol,
)
from cywl_oopz.features.agent.selection import ProviderSelectionService
from cywl_oopz.features.agent.service import AgentConversationService
from cywl_oopz.features.agent.skills.availability import SkillAvailabilityService
from cywl_oopz.features.agent.skills.catalog import ReloadableAgentSkillCatalog
from cywl_oopz.features.agent.skills.models import AgentSkill
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind
from cywl_oopz.settings import AgentMode, AgentSettings

PROVIDER_ID = UUID("10000000-0000-0000-0000-000000000001")
DEFAULT_MODEL_ID = UUID("10000000-0000-0000-0000-000000000002")
OTHER_MODEL_ID = UUID("10000000-0000-0000-0000-000000000003")


class InMemoryCatalogRepository:
    def __init__(self) -> None:
        self.provider = LlmProvider(
            id=PROVIDER_ID,
            alias="provider",
            display_name="Provider",
            protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
            base_url="https://llm.example/v1",
            api_key="database-key",
            user_selectable=True,
            enabled=True,
        )
        self.models = (
            LlmModel(
                id=DEFAULT_MODEL_ID,
                provider_id=PROVIDER_ID,
                alias="default",
                remote_model_name="default-remote",
                display_name="Default",
                enabled=True,
                is_provider_default=True,
                is_application_default=True,
                capabilities=frozenset({ModelCapability.TOOL_CALLING}),
            ),
            LlmModel(
                id=OTHER_MODEL_ID,
                provider_id=PROVIDER_ID,
                alias="other",
                remote_model_name="other-remote",
                display_name="Other",
                enabled=True,
                is_provider_default=False,
                is_application_default=False,
                capabilities=frozenset({ModelCapability.TOOL_CALLING}),
            ),
        )

    async def load_providers(self) -> tuple[LlmProvider, ...]:
        return (self.provider,)

    async def load_models(self) -> tuple[LlmModel, ...]:
        return self.models


class InMemoryThreads:
    def __init__(self) -> None:
        self.values: dict[ConversationKey, AgentThread] = {}

    async def get(self, key: ConversationKey) -> AgentThread | None:
        return self.values.get(key)

    async def add(self, thread: AgentThread) -> None:
        self.values[thread.key] = thread

    async def set_selected_model(self, key: ConversationKey, model_id: UUID) -> None:
        self.values[key] = replace(self.values[key], selected_model_id=model_id)

    async def refresh_expiry(self, thread_id: UUID, expires_at) -> None:
        key = next(key for key, thread in self.values.items() if thread.id == thread_id)
        self.values[key] = replace(self.values[key], expires_at=expires_at)

    async def save_summary(
        self,
        thread_id: UUID,
        summary: str,
        through_sequence: int,
        *,
        expected_version: int,
    ) -> bool:
        key = next(key for key, thread in self.values.items() if thread.id == thread_id)
        thread = self.values[key]
        if thread.version != expected_version:
            return False
        self.values[key] = replace(
            thread,
            summary=summary,
            summary_through_sequence=through_sequence,
            summary_version=thread.summary_version + 1,
            version=thread.version + 1,
        )
        return True

    async def delete(self, key: ConversationKey) -> None:
        self.values.pop(key, None)


class InMemorySelections:
    def __init__(self, threads: InMemoryThreads) -> None:
        self.threads = threads
        self.user_models: dict[str, UUID] = {}

    async def load_candidates(self, key: ConversationKey) -> ModelSelectionCandidates:
        thread = self.threads.values.get(key)
        return ModelSelectionCandidates(
            thread_model_id=thread.selected_model_id if thread is not None else None,
            user_model_id=self.user_models.get(key.person_id),
            application_model_id=DEFAULT_MODEL_ID,
        )

    async def set_user_model(self, person_id: str, model_id: UUID) -> None:
        self.user_models[person_id] = model_id


class InMemoryRuns:
    def __init__(self) -> None:
        self.runs: dict[UUID, AgentRun] = {}
        self.states: dict[UUID, AgentRunState] = {}

    async def add(self, run: AgentRun) -> None:
        self.runs[run.id] = run
        self.states[run.id] = run.state

    async def finish(
        self,
        state: AgentRunState,
        *,
        usage: dict[str, object],
        error_code: str = "",
    ) -> None:
        self.states[state.run_id] = state

    async def abandon_stale(self, before, now) -> int:
        return 0


class InMemoryMessages:
    def __init__(self) -> None:
        self.values: dict[UUID, list[AgentMessage]] = {}

    async def load(
        self,
        thread_id: UUID,
        *,
        limit: int,
        after_sequence: int = 0,
    ) -> tuple[AgentMessage, ...]:
        messages = [
            item for item in self.values.get(thread_id, []) if (item.sequence or 0) > after_sequence
        ]
        return tuple(messages[-limit:])

    async def load_after(
        self,
        thread_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> tuple[AgentMessage, ...]:
        return tuple(
            item for item in self.values.get(thread_id, []) if (item.sequence or 0) > after_sequence
        )[:limit]

    async def append(
        self,
        thread_id: UUID,
        run_id: UUID,
        messages: tuple[AgentMessage, ...],
    ) -> None:
        stored = self.values.setdefault(thread_id, [])
        stored.extend(
            replace(message, sequence=len(stored) + offset)
            for offset, message in enumerate(messages, start=1)
        )

    async def count(self, thread_id: UUID) -> int:
        return len(self.values.get(thread_id, []))


class RecordingEngine:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = outputs or ["answer"]
        self.requests = []
        self.progress = []

    async def run(self, request, progress=None):
        self.requests.append(request)
        self.progress.append(progress)
        return AgentRunResult(
            output=self.outputs.pop(0),
            stop_reason=AgentStopReason.COMPLETED,
            input_tokens=4,
            output_tokens=2,
            model_requests=1,
            tool_calls=2,
        )

    async def aclose(self) -> None:
        return None


class FailingEngine(RecordingEngine):
    async def run(self, request, progress=None):
        del progress
        self.requests.append(request)
        raise ProviderError("provider unavailable")


class InMemorySkillRepository:
    def __init__(self, skills: tuple[AgentSkill, ...]) -> None:
        self.skills = skills

    async def generation(self) -> int:
        return 1

    async def load_enabled(self) -> tuple[AgentSkill, ...]:
        return self.skills


class StaticToolAvailability:
    async def names(self, identity, model) -> tuple[str, ...]:
        del identity, model
        return ("load_agent_skill", "read_agent_skill_resource")


def agent_settings() -> AgentSettings:
    return AgentSettings(
        mode=AgentMode.AGENT,
        system_prompt="Agent system prompt",
        live_display=False,
        display_edit_interval_seconds=0.8,
        session_ttl_seconds=3600,
        max_history_messages=10,
        max_history_characters=1000,
        timeout_seconds=5,
        max_model_requests=3,
        max_tool_calls=2,
        max_total_tokens=1000,
        max_parallel_tools=1,
        enabled_tools=(),
        tool_timeout_seconds=1,
        max_tool_result_characters=1000,
        summary_enabled=True,
        summary_trigger_messages=6,
        summary_retain_messages=2,
        summary_timeout_seconds=1,
        summary_max_characters=1000,
        memory_enabled_by_default=True,
        memory_default_ttl_days=30,
        memory_max_items=20,
        memory_context_items=6,
        memory_max_item_characters=1000,
        stale_run_after_seconds=30,
    )


async def build_service(
    chat_settings,
    engine=None,
    *,
    tool_availability=None,
    skill_catalog=None,
    skill_availability=None,
):
    catalog = ReloadableProviderCatalog(InMemoryCatalogRepository())
    await catalog.reload()
    threads = InMemoryThreads()
    selections = InMemorySelections(threads)
    runs = InMemoryRuns()
    messages = InMemoryMessages()
    service = AgentConversationService(
        agent_settings(),
        chat_settings,
        engine or RecordingEngine(),
        catalog,
        ProviderSelectionService(catalog, selections),
        selections,
        threads,
        runs,
        messages,
        tool_availability,
        skill_catalog,
        skill_availability,
    )
    return service, threads, selections, runs, messages


def key(person: str = "person") -> ConversationKey:
    return ConversationKey("channel", "area", "channel", person)


@pytest.mark.asyncio
async def test_agent_service_persists_turns_and_reuses_provider_neutral_history(
    chat_settings,
) -> None:
    engine = RecordingEngine(["first answer", "second answer"])
    service, threads, _, runs, messages = await build_service(chat_settings, engine)

    first = await service.ask(key(), "first question")
    second = await service.ask(key(), "second question")

    assert first.content == "first answer"
    assert first.elapsed_seconds is not None
    assert first.elapsed_seconds >= 0
    assert first.input_tokens == 4
    assert first.output_tokens == 2
    assert first.model_requests == 1
    assert first.tool_calls == 2
    assert second.model == "provider/default"
    assert [message.role for message in engine.requests[1].context] == [
        "system",
        "user",
        "assistant",
    ]
    thread = threads.values[key()]
    assert [message.content["text"] for message in messages.values[thread.id]] == [
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]
    assert all(state.status is AgentRunStatus.SUCCEEDED for state in runs.states.values())


@pytest.mark.asyncio
async def test_agent_service_pins_skill_scope_and_hides_loaders_for_empty_catalog(
    chat_settings,
) -> None:
    skill = AgentSkill(
        id=uuid4(),
        name="web-research",
        display_name="网页研究",
        description="Research current facts.",
        instructions="Search and read sources.",
        version="1",
        revision=1,
        required_tools=frozenset(),
        resources=(),
        metadata={},
    )
    catalog = ReloadableAgentSkillCatalog(
        InMemorySkillRepository((skill,)),
        registered_tools=("load_agent_skill", "read_agent_skill_resource"),
        refresh_seconds=30,
        max_available_skills=8,
    )
    await catalog.reload()
    engine = RecordingEngine()
    service = (
        await build_service(
            chat_settings,
            engine,
            tool_availability=StaticToolAvailability(),
            skill_catalog=catalog,
            skill_availability=SkillAvailabilityService(),
        )
    )[0]
    await service.ask(key(), "research this")

    assert engine.requests[0].enabled_tools == (
        "load_agent_skill",
        "read_agent_skill_resource",
    )
    assert engine.requests[0].skill_scope is not None
    assert [item.name for item in engine.requests[0].skill_scope.available_skills] == [
        "web-research"
    ]

    empty_catalog = ReloadableAgentSkillCatalog(
        InMemorySkillRepository(()),
        registered_tools=("load_agent_skill", "read_agent_skill_resource"),
        refresh_seconds=30,
        max_available_skills=8,
    )
    await empty_catalog.reload()
    empty_engine = RecordingEngine()
    empty_service = (
        await build_service(
            chat_settings,
            empty_engine,
            tool_availability=StaticToolAvailability(),
            skill_catalog=empty_catalog,
            skill_availability=SkillAvailabilityService(),
        )
    )[0]
    await empty_service.ask(key("other"), "ordinary chat")

    assert empty_engine.requests[0].enabled_tools == ()
    assert empty_engine.requests[0].skill_scope is None


@pytest.mark.asyncio
async def test_agent_service_emits_accepted_and_passes_the_same_progress_sink(
    chat_settings,
) -> None:
    class RecordingProgress:
        def __init__(self) -> None:
            self.events: list[ConversationProgressEvent] = []

        async def emit(self, event: ConversationProgressEvent) -> None:
            self.events.append(event)

    engine = RecordingEngine()
    progress = RecordingProgress()
    service, _, _, _, _ = await build_service(chat_settings, engine)

    await service.ask(key(), "question", progress=progress)

    assert [event.kind for event in progress.events] == [ProgressKind.ACCEPTED]
    assert engine.progress == [progress]


@pytest.mark.asyncio
async def test_agent_service_switches_thread_and_user_provider_preferences(
    chat_settings,
) -> None:
    service, _, selections, _, _ = await build_service(chat_settings)

    selected = await service.select_provider(
        key(),
        "provider",
        "other",
        user_default=False,
    )
    current = await service.current_selection(key())
    await service.select_provider(
        key("another-person"),
        "provider",
        "other",
        user_default=True,
    )

    assert selected == "provider/other"
    assert current.model.model_id == OTHER_MODEL_ID
    assert selections.user_models["another-person"] == OTHER_MODEL_ID
    assert service.list_models() == ("provider/default", "provider/other")


@pytest.mark.asyncio
async def test_agent_service_records_provider_failure_without_assistant_message(
    chat_settings,
) -> None:
    service, threads, _, runs, messages = await build_service(
        chat_settings,
        FailingEngine(),
    )

    with pytest.raises(ProviderError):
        await service.ask(key(), "question")

    thread = threads.values[key()]
    assert [message.role for message in messages.values[thread.id]] == ["user"]
    assert next(iter(runs.states.values())).stop_reason is AgentStopReason.PROVIDER_ERROR


@pytest.mark.asyncio
async def test_agent_service_cancellation_marks_run_and_releases_lock(chat_settings) -> None:
    started = asyncio.Event()

    class WaitingEngine(RecordingEngine):
        async def run(self, request, progress=None):
            del progress
            self.requests.append(request)
            started.set()
            await asyncio.Event().wait()

    engine = WaitingEngine()
    service, _, _, runs, _ = await build_service(chat_settings, engine)
    task = asyncio.create_task(service.ask(key(), "wait"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    state = next(iter(runs.states.values()))
    assert state.status is AgentRunStatus.CANCELLED
    assert state.stop_reason is AgentStopReason.CANCELLED


@pytest.mark.asyncio
async def test_agent_service_persists_and_reuses_only_paired_tool_messages(
    chat_settings,
) -> None:
    pair = (
        AgentMessage(
            "assistant",
            "tool_call",
            {
                "version": 1,
                "tool_call_id": "call-1",
                "tool_name": "get_agent_status",
                "arguments": {},
            },
        ),
        AgentMessage(
            "tool",
            "tool_result",
            {
                "version": 1,
                "tool_call_id": "call-1",
                "tool_name": "get_agent_status",
                "result": {"ok": True},
            },
        ),
    )

    class PairingEngine(RecordingEngine):
        async def run(self, request, progress=None):
            del progress
            self.requests.append(request)
            return AgentRunResult(
                output="answer",
                stop_reason=AgentStopReason.COMPLETED,
                intermediate_messages=pair if len(self.requests) == 1 else (),
            )

    engine = PairingEngine()
    service, threads, _, _, messages = await build_service(chat_settings, engine)

    await service.ask(key(), "first")
    await service.ask(key(), "second")

    assert [item.kind for item in engine.requests[1].context] == [
        "text",
        "text",
        "tool_call",
        "tool_result",
        "text",
    ]
    thread = threads.values[key()]
    assert [item.kind for item in messages.values[thread.id]][:4] == [
        "text",
        "tool_call",
        "tool_result",
        "text",
    ]

    unpaired = (
        AgentMessage(
            "tool",
            "tool_result",
            {
                "tool_call_id": "orphan",
                "tool_name": "missing",
                "result": {},
            },
        ),
        AgentMessage("assistant", "text", {"text": "orphan answer"}),
    )
    assert service._context_builder.trim_history(unpaired) == ()
