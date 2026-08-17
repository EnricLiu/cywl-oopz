from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from cywl_oopz.core.errors import (
    AgentInternalError,
    DatabaseError,
    ProviderError,
    ProviderTimeoutError,
)
from cywl_oopz.core.lifecycle import ModelSelectionSource
from cywl_oopz.features.agent.models import (
    AgentIdentity,
    AgentMessage,
    AgentModelRef,
    AgentRun,
    AgentRunLimits,
    AgentRunResult,
    AgentRunState,
    AgentRunStatus,
    AgentStopReason,
    AgentThread,
    ModelCapability,
    ProviderProtocol,
)
from cywl_oopz.features.agent.run_service import AgentRunService, AgentRunSpec
from cywl_oopz.features.chat.models import ConversationKey


class RecordingRuns:
    def __init__(self) -> None:
        self.runs: dict[UUID, AgentRun] = {}
        self.states: dict[UUID, AgentRunState] = {}
        self.usages: dict[UUID, dict[str, object]] = {}
        self.errors: dict[UUID, str] = {}
        self.heartbeats: list[tuple[UUID, datetime]] = []

    async def add(self, run: AgentRun) -> None:
        self.runs[run.id] = run
        self.states[run.id] = run.state

    async def finish(self, state, *, usage, error_code="") -> None:
        self.states[state.run_id] = state
        self.usages[state.run_id] = dict(usage)
        self.errors[state.run_id] = error_code

    async def heartbeat(self, run_id: UUID, now: datetime) -> bool:
        self.heartbeats.append((run_id, now))
        return self.states[run_id].status is AgentRunStatus.RUNNING

    async def abandon_stale(self, before, now) -> int:
        del before, now
        return 0


class RecordingMessages:
    def __init__(self) -> None:
        self.values: dict[UUID, list[AgentMessage]] = {}

    async def append(self, thread_id, run_id, messages) -> None:
        del run_id
        self.values.setdefault(thread_id, []).extend(messages)


@dataclass
class ScriptedEngine:
    result: AgentRunResult | BaseException
    delay_seconds: float = 0

    def __post_init__(self) -> None:
        self.requests = []
        self.progress = []

    async def run(self, request, progress=None):
        self.requests.append(request)
        self.progress.append(progress)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def aclose(self) -> None:
        return None


def run_spec() -> AgentRunSpec:
    key = ConversationKey("delegated_task", "area", "text", "person")
    model = AgentModelRef(
        uuid4(),
        uuid4(),
        "provider",
        "model",
        "remote-model",
        ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
        frozenset({ModelCapability.TOOL_CALLING}),
        None,
    )
    return AgentRunSpec(
        thread=AgentThread(uuid4(), key, None, datetime.now(UTC) + timedelta(hours=1)),
        identity=AgentIdentity("person", key, transport_channel_id="text"),
        prompt="完成后台查询",
        model=model,
        selection_source=ModelSelectionSource.USER,
        enabled_tools=("search_web", "read_web_page"),
        limits=AgentRunLimits(timeout_seconds=2),
        context=(AgentMessage("system", "text", {"text": "system"}),),
    )


@pytest.mark.asyncio
async def test_agent_run_service_persists_isolated_run_messages_usage_and_heartbeat() -> None:
    intermediate = AgentMessage(
        "assistant",
        "tool_call",
        {
            "version": 1,
            "tool_call_id": "call-1",
            "tool_name": "search_web",
            "arguments": {"query": "Miku"},
        },
    )
    result = AgentRunResult(
        "找到结果",
        AgentStopReason.COMPLETED,
        input_tokens=8,
        output_tokens=3,
        model_requests=2,
        tool_calls=1,
        intermediate_messages=(intermediate,),
    )
    engine = ScriptedEngine(result, delay_seconds=0.035)
    runs = RecordingRuns()
    messages = RecordingMessages()
    service = AgentRunService(
        engine,
        runs,
        messages,
        heartbeat_interval_seconds=0.01,
    )
    spec = run_spec()

    outcome = await service.run(spec)

    assert outcome.result is result
    assert outcome.run_id == engine.requests[0].run_id
    assert engine.requests[0].identity.conversation.scope == "delegated_task"
    assert engine.requests[0].enabled_tools == ("search_web", "read_web_page")
    assert [item.kind for item in messages.values[spec.thread.id]] == [
        "text",
        "tool_call",
        "text",
    ]
    assert runs.states[outcome.run_id].status is AgentRunStatus.SUCCEEDED
    assert runs.usages[outcome.run_id] == {
        "input_tokens": 8,
        "output_tokens": 3,
        "model_requests": 2,
        "tool_calls": 1,
    }
    assert len(runs.heartbeats) >= 2


@pytest.mark.asyncio
async def test_agent_run_service_binds_progress_after_run_persistence_before_engine() -> None:
    events: list[tuple[str, UUID]] = []

    class RecordingProgress:
        async def bind_run(self, run_id: UUID) -> None:
            assert run_id in runs.runs
            events.append(("bound", run_id))

        async def emit(self, event) -> None:
            del event

    class AssertingEngine(ScriptedEngine):
        async def run(self, request, progress=None):
            assert events == [("bound", request.run_id)]
            return await super().run(request, progress)

    result = AgentRunResult("完成", AgentStopReason.COMPLETED)
    runs = RecordingRuns()
    service = AgentRunService(AssertingEngine(result), runs, RecordingMessages())

    outcome = await service.run(run_spec(), RecordingProgress())

    assert events == [("bound", outcome.run_id)]


@pytest.mark.parametrize(
    ("error", "reason", "code"),
    (
        (ProviderTimeoutError("timeout"), AgentStopReason.TIMEOUT, "provider_timeout"),
        (ProviderError("failed"), AgentStopReason.PROVIDER_ERROR, "provider_error"),
        (AgentInternalError("adapter"), AgentStopReason.INVALID_OUTPUT, "agent_internal"),
        (RuntimeError("bad output"), AgentStopReason.INVALID_OUTPUT, "agent_error"),
    ),
)
@pytest.mark.asyncio
async def test_agent_run_service_classifies_terminal_failures(error, reason, code) -> None:
    runs = RecordingRuns()
    messages = RecordingMessages()
    service = AgentRunService(ScriptedEngine(error), runs, messages)
    spec = run_spec()

    with pytest.raises(type(error)):
        await service.run(spec)

    run_id = next(iter(runs.states))
    assert runs.states[run_id].stop_reason is reason
    assert runs.errors[run_id] == code
    assert [message.role for message in messages.values[spec.thread.id]] == ["user"]


@pytest.mark.asyncio
async def test_agent_run_service_cancellation_is_terminal_and_stops_heartbeat() -> None:
    started = asyncio.Event()

    class WaitingEngine(ScriptedEngine):
        async def run(self, request, progress=None):
            del request, progress
            started.set()
            await asyncio.Event().wait()

    runs = RecordingRuns()
    service = AgentRunService(
        WaitingEngine(AgentRunResult("unused", AgentStopReason.COMPLETED)),
        runs,
        RecordingMessages(),
        heartbeat_interval_seconds=0.01,
    )
    task = asyncio.create_task(service.run(run_spec()))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    run_id = next(iter(runs.states))
    assert runs.states[run_id].status is AgentRunStatus.CANCELLED
    heartbeat_count = len(runs.heartbeats)
    await asyncio.sleep(0.02)
    assert len(runs.heartbeats) == heartbeat_count


@pytest.mark.asyncio
async def test_agent_run_service_finishes_run_when_user_message_append_fails() -> None:
    class UserAppendFailure(RecordingMessages):
        async def append(self, thread_id, run_id, messages) -> None:
            del thread_id, run_id, messages
            raise DatabaseError("user append unavailable")

    runs = RecordingRuns()
    service = AgentRunService(
        ScriptedEngine(AgentRunResult("unused", AgentStopReason.COMPLETED)),
        runs,
        UserAppendFailure(),
    )

    with pytest.raises(DatabaseError, match="user append unavailable"):
        await service.run(run_spec())

    run_id = next(iter(runs.states))
    assert runs.states[run_id].stop_reason is AgentStopReason.INVALID_OUTPUT
    assert runs.errors[run_id] == "user_message_persistence_error"


@pytest.mark.asyncio
async def test_agent_run_service_returns_generated_answer_when_result_append_fails() -> None:
    class ResultAppendFailure(RecordingMessages):
        def __init__(self) -> None:
            super().__init__()
            self.append_count = 0

        async def append(self, thread_id, run_id, messages) -> None:
            self.append_count += 1
            if self.append_count == 2:
                raise DatabaseError("result append unavailable")
            await super().append(thread_id, run_id, messages)

    result = AgentRunResult("generated answer", AgentStopReason.COMPLETED)
    runs = RecordingRuns()
    messages = ResultAppendFailure()
    spec = run_spec()

    outcome = await AgentRunService(
        ScriptedEngine(result),
        runs,
        messages,
    ).run(spec)

    assert outcome.result is result
    assert outcome.persistence_degraded is True
    assert [message.role for message in messages.values[spec.thread.id]] == ["user"]
    assert runs.states[outcome.run_id].stop_reason is AgentStopReason.INVALID_OUTPUT
    assert runs.errors[outcome.run_id] == "result_message_persistence_error"


@pytest.mark.asyncio
async def test_agent_run_service_returns_generated_answer_when_run_finish_fails() -> None:
    class FinishFailureRuns(RecordingRuns):
        async def finish(self, state, *, usage, error_code="") -> None:
            del state, usage, error_code
            raise DatabaseError("run finish unavailable")

    result = AgentRunResult("generated answer", AgentStopReason.COMPLETED)
    runs = FinishFailureRuns()
    messages = RecordingMessages()
    spec = run_spec()

    outcome = await AgentRunService(
        ScriptedEngine(result),
        runs,
        messages,
    ).run(spec)

    assert outcome.result is result
    assert outcome.persistence_degraded is True
    assert [message.role for message in messages.values[spec.thread.id]] == [
        "user",
        "assistant",
    ]
    assert runs.states[outcome.run_id].status is AgentRunStatus.RUNNING


@pytest.mark.asyncio
async def test_agent_run_service_cleanup_failure_preserves_provider_error() -> None:
    class FinishFailureRuns(RecordingRuns):
        async def finish(self, state, *, usage, error_code="") -> None:
            del state, usage, error_code
            raise RuntimeError("cleanup failed")

    provider_error = ProviderError("primary provider failure")
    runs = FinishFailureRuns()
    service = AgentRunService(
        ScriptedEngine(provider_error),
        runs,
        RecordingMessages(),
    )

    with pytest.raises(ProviderError) as captured:
        await service.run(run_spec())

    assert captured.value is provider_error
