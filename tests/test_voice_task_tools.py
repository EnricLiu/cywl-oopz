from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from time import monotonic
from uuid import UUID, uuid4

import pytest

from cywl_oopz.features.agent.delegation.models import (
    CancelOutcome,
    DelegatedAgentTask,
    DelegatedResultStyle,
    DelegatedTaskLane,
    DelegatedTaskNotificationState,
    DelegatedTaskPolicy,
    DelegatedTaskStatus,
    TaskRef,
)
from cywl_oopz.features.agent.delegation.service import VoiceDelegatedTaskService
from cywl_oopz.features.voice.models import (
    VoiceChannelKey,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.task_tools import VoiceTaskControlTools


class MemoryTaskRepository:
    def __init__(self) -> None:
        self.model_id = uuid4()
        self.profile = "voice_readonly_v1"
        self.tasks: list[DelegatedAgentTask] = []

    async def resolve_submission_policy(self, session_id, owner_person_id):
        del session_id, owner_person_id
        return DelegatedTaskPolicy(self.profile, self.model_id)

    async def submit(self, request):
        for task in self.tasks:
            if (
                task.origin_voice_session_id == request.origin_voice_session_id
                and task.provider_call_id == request.provider_call_id
            ):
                return task
        now = datetime.now(UTC)
        task = DelegatedAgentTask(
            id=uuid4(),
            owner_person_id=request.owner_person_id,
            area_id=request.area_id,
            text_channel_id=request.text_channel_id,
            voice_channel_id=request.voice_channel_id,
            origin_voice_session_id=request.origin_voice_session_id,
            session_sequence=len(self.tasks) + 1,
            provider_call_id=request.provider_call_id,
            objective=request.objective,
            result_style=request.result_style,
            status=DelegatedTaskStatus.QUEUED,
            lane=request.lane,
            conflict_key=request.conflict_key,
            notification_state=DelegatedTaskNotificationState.PENDING,
            agent_model_id=request.agent_model_id,
            allowed_tool_names=request.allowed_tool_names,
            created_at=now,
            updated_at=now,
        )
        self.tasks.append(task)
        return task

    async def get_for_owner(self, task_ref: TaskRef, owner_person_id: str):
        for task in self.tasks:
            if task.owner_person_id != owner_person_id:
                continue
            if task_ref.task_id == task.id:
                return task
            if (
                task_ref.origin_voice_session_id == task.origin_voice_session_id
                and task_ref.session_sequence == task.session_sequence
            ):
                return task
        return None

    async def list_for_owner(self, owner_person_id, query):
        return tuple(
            task
            for task in reversed(self.tasks)
            if task.owner_person_id == owner_person_id
            and (query.status is None or task.status is query.status)
            and (
                query.origin_voice_session_id is None
                or task.origin_voice_session_id == query.origin_voice_session_id
            )
        )[: query.limit]

    async def request_cancel(self, task_id: UUID, owner_person_id: str):
        for index, task in enumerate(self.tasks):
            if task.id == task_id and task.owner_person_id == owner_person_id:
                cancelled = replace(
                    task,
                    status=DelegatedTaskStatus.CANCELLED,
                    cancel_requested_at=datetime.now(UTC),
                    finished_at=datetime.now(UTC),
                )
                self.tasks[index] = cancelled
                return CancelOutcome(cancelled, True)
        return CancelOutcome(None, False)


class DeterministicFakeRunner:
    """Accept wakeups immediately while a test-controlled background job remains pending."""

    def __init__(self) -> None:
        self.woken: list[UUID] = []
        self.release = asyncio.Event()
        self.finished: list[UUID] = []
        self.tasks: set[asyncio.Task[None]] = set()

    async def wake(self, task_id: UUID) -> None:
        self.woken.append(task_id)
        task = asyncio.create_task(self._run(task_id))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _run(self, task_id: UUID) -> None:
        await self.release.wait()
        self.finished.append(task_id)

    async def aclose(self) -> None:
        self.release.set()
        await asyncio.gather(*self.tasks)


class RecordingCompletionNotifier:
    def __init__(self) -> None:
        self.owners: list[str] = []

    async def wake(self, owner_person_id: str) -> None:
        self.owners.append(owner_person_id)


def descriptor(owner: str = "person") -> VoiceSessionDescriptor:
    return VoiceSessionDescriptor(
        uuid4(),
        owner,
        VoiceChannelKey("area", "voice"),
        VoiceTextAddress("area", "text"),
    )


def test_voice_task_schemas_are_short_and_do_not_expose_server_policy() -> None:
    schemas = VoiceTaskControlTools.schemas()

    assert {schema["name"] for schema in schemas} == {
        "delegate_agent_task",
        "get_agent_task",
        "list_agent_tasks",
        "read_agent_task_result",
        "cancel_agent_task",
    }
    encoded = str(schemas)
    for forbidden in (
        "owner_person_id",
        "session_id",
        "model_id",
        "allowed_tools",
        "conflict_key",
        "parallel",
    ):
        assert forbidden not in encoded


@pytest.mark.asyncio
async def test_delegate_is_durable_idempotent_owner_scoped_and_nonblocking() -> None:
    repository = MemoryTaskRepository()
    runner = DeterministicFakeRunner()
    completion_notifier = RecordingCompletionNotifier()
    tools = VoiceTaskControlTools(
        VoiceDelegatedTaskService(
            repository,
            runner,
            completion_notifier=completion_notifier,
        )
    )
    session = descriptor()

    started = monotonic()
    first = await tools.execute(
        session,
        "call-1",
        "delegate_agent_task",
        {"objective": "搜索最近的初音未来演出"},
    )
    elapsed = monotonic() - started
    duplicate = await tools.execute(
        session,
        "call-1",
        "delegate_agent_task",
        {"objective": "搜索最近的初音未来演出"},
    )

    assert elapsed < 0.25
    assert first["accepted"] is True
    assert first["task"] == duplicate["task"] == "T1"
    assert len(repository.tasks) == 1
    assert runner.finished == []
    assert set(repository.tasks[0].allowed_tool_names).isdisjoint(
        {
            "delegate_agent_task",
            "react_to_message",
            "browser_click",
            "create_agent_skill",
            "import_netease_playlist",
        }
    )

    intruder = replace(session, owner_person_id="another-person")
    assert await tools.execute(intruder, "read-1", "get_agent_task", {"task": "T1"}) == {
        "ok": False,
        "code": "task_not_found",
    }
    listed = await tools.execute(session, "list-1", "list_agent_tasks", {})
    assert listed["tasks"][0]["task"] == "T1"
    cancelled = await tools.execute(session, "cancel-1", "cancel_agent_task", {"task": "T1"})
    assert cancelled["cancel_requested"] is True
    assert completion_notifier.owners == ["person"]

    runner.release.set()
    await runner.aclose()
    assert runner.finished


@pytest.mark.asyncio
async def test_task_control_validation_returns_stable_bounded_errors() -> None:
    repository = MemoryTaskRepository()
    runner = DeterministicFakeRunner()
    tools = VoiceTaskControlTools(VoiceDelegatedTaskService(repository, runner))
    session = descriptor()

    assert await tools.execute(session, "x", "unknown", {}) == {
        "ok": False,
        "code": "tool_not_allowed",
    }
    invalid = await tools.execute(
        session,
        "x",
        "delegate_agent_task",
        {"objective": "", "model_id": str(uuid4())},
    )
    assert invalid == {"ok": False, "code": "invalid_arguments"}
    assert (await tools.execute(session, "x", "read_agent_task_result", {"task": "T404"})) == {
        "ok": False,
        "code": "task_not_found",
    }
    await runner.aclose()


@pytest.mark.asyncio
async def test_mutation_profile_is_server_owned_area_serial_and_excludes_media_control() -> None:
    repository = MemoryTaskRepository()
    repository.profile = "voice_mutation_v1"
    runner = DeterministicFakeRunner()
    service = VoiceDelegatedTaskService(repository, runner)
    session = descriptor()

    created = await service.delegate(
        session,
        "mutation-call",
        "新建一个共享歌单并加入两首歌",
        result_style=DelegatedResultStyle.BRIEF,
    )

    assert created.lane is DelegatedTaskLane.MUTATION_SERIAL
    assert created.conflict_key == "area:area"
    assert {
        "create_music_playlist",
        "add_music_playlist_track",
        "rename_music_playlist",
        "delete_music_playlist",
        "clear_music_playlist",
        "create_agent_skill",
        "update_agent_skill",
    }.issubset(created.allowed_tool_names)
    assert {
        "enqueue_music",
        "load_music_playlist",
        "set_music_playback_mode",
        "clear_music_queue",
        "delegate_agent_task",
    }.isdisjoint(created.allowed_tool_names)
    runner.release.set()
    await runner.aclose()
