"""Business rules for fast realtime delegated-task control operations."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.voice.models import VoiceSessionDescriptor

from .models import (
    CancelOutcome,
    DelegatedAgentTask,
    DelegatedResultStyle,
    DelegatedTaskLane,
    DelegatedTaskStatus,
    DelegatedTaskSubmission,
    TaskListQuery,
    TaskRef,
)
from .ports import (
    DelegatedTaskCompletionNotifier,
    DelegatedTaskRepository,
    DelegatedTaskWakeup,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DelegatedTaskProfile:
    name: str
    lane: DelegatedTaskLane
    allowed_tool_names: tuple[str, ...]
    conflict_key: str = ""

    def resolve_conflict_key(self, descriptor: VoiceSessionDescriptor) -> str:
        if not self.conflict_key:
            return ""
        if self.conflict_key == "area":
            return f"area:{descriptor.voice_channel.area_id}"
        raise ValueError("Delegated task profile conflict scope is unsupported")


class DelegatedTaskProfileCatalog:
    """Resolve server-owned task capabilities; model arguments cannot alter them."""

    READONLY_V1 = DelegatedTaskProfile(
        name="voice_readonly_v1",
        lane=DelegatedTaskLane.READ_PARALLEL,
        allowed_tool_names=(
            "get_agent_status",
            "get_channel_settings",
            "get_music_queue",
            "get_music_playlist",
            "inspect_agent_skill",
            "list_agent_skill_library",
            "list_music_playlists",
            "load_agent_skill",
            "preview_netease_playlist",
            "read_agent_skill_resource",
            "read_web_page",
            "search_music_catalog",
            "search_web",
        ),
    )
    MUTATION_V1 = DelegatedTaskProfile(
        name="voice_mutation_v1",
        lane=DelegatedTaskLane.MUTATION_SERIAL,
        allowed_tool_names=(
            *READONLY_V1.allowed_tool_names,
            "add_music_playlist_track",
            "create_agent_skill",
            "create_music_playlist",
            "import_netease_playlist",
            "invite_agent_skill_share",
            "manage_agent_skill_resource",
            "remove_music_playlist_track",
            "respond_agent_skill_share",
            "revoke_agent_skill_share",
            "set_agent_skill_state",
            "update_agent_skill",
        ),
        conflict_key="area",
    )

    def __init__(self) -> None:
        self._profiles = MappingProxyType(
            {
                self.READONLY_V1.name: self.READONLY_V1,
                self.MUTATION_V1.name: self.MUTATION_V1,
            }
        )

    def resolve(self, name: str) -> DelegatedTaskProfile:
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise ValueError("Delegated task profile is not supported") from exc


class InProcessDelegatedTaskWakeup:
    """Lossy low-latency signal; PostgreSQL remains the durable queue."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._task_ids: set[UUID] = set()
        self._lock = asyncio.Lock()

    async def wake(self, task_id: UUID) -> None:
        async with self._lock:
            self._task_ids.add(task_id)
            self._event.set()

    async def wait(self, timeout_seconds: float) -> tuple[UUID, ...]:
        if timeout_seconds <= 0:
            raise ValueError("Delegated task wake timeout must be positive")
        try:
            async with asyncio.timeout(timeout_seconds):
                await self._event.wait()
        except TimeoutError:
            return ()
        async with self._lock:
            task_ids = tuple(self._task_ids)
            self._task_ids.clear()
            self._event.clear()
            return task_ids


class VoiceDelegatedTaskService:
    """Owner-bound facade shared by realtime tools and future text commands."""

    def __init__(
        self,
        repository: DelegatedTaskRepository,
        wakeup: DelegatedTaskWakeup,
        profiles: DelegatedTaskProfileCatalog | None = None,
        completion_notifier: DelegatedTaskCompletionNotifier | None = None,
    ) -> None:
        self._repository = repository
        self._wakeup = wakeup
        self._profiles = profiles or DelegatedTaskProfileCatalog()
        self._completion_notifier = completion_notifier

    async def delegate(
        self,
        descriptor: VoiceSessionDescriptor,
        provider_call_id: str,
        objective: str,
        result_style: DelegatedResultStyle,
    ) -> DelegatedAgentTask:
        normalized = " ".join(objective.split())
        if not normalized or len(normalized) > 2000:
            raise ValueError("Task objective must contain 1-2000 characters")
        call_id = provider_call_id.strip()
        if not call_id or len(call_id) > 256:
            raise ValueError("Provider call identifier is invalid")
        policy = await self._repository.resolve_submission_policy(
            descriptor.session_id,
            descriptor.owner_person_id,
        )
        profile = self._profiles.resolve(policy.profile)
        task = await self._repository.submit(
            DelegatedTaskSubmission(
                owner_person_id=descriptor.owner_person_id,
                area_id=descriptor.voice_channel.area_id,
                text_channel_id=descriptor.origin.channel_id,
                voice_channel_id=descriptor.voice_channel.channel_id,
                origin_voice_session_id=descriptor.session_id,
                provider_call_id=call_id,
                objective=normalized,
                result_style=result_style,
                lane=profile.lane,
                conflict_key=profile.resolve_conflict_key(descriptor),
                agent_model_id=policy.agent_model_id,
                allowed_tool_names=profile.allowed_tool_names,
            )
        )
        await self._wakeup.wake(task.id)
        logger.info(
            "Delegated Agent task accepted: task=%s session=%s alias=%s lane=%s",
            opaque_ref(str(task.id)),
            opaque_ref(str(descriptor.session_id)),
            task.alias,
            task.lane.value,
        )
        return task

    async def get(
        self,
        descriptor: VoiceSessionDescriptor,
        task: str,
    ) -> DelegatedAgentTask | None:
        return await self._repository.get_for_owner(
            TaskRef.parse(task, origin_voice_session_id=descriptor.session_id),
            descriptor.owner_person_id,
        )

    async def list(
        self,
        descriptor: VoiceSessionDescriptor,
        *,
        status: DelegatedTaskStatus | None = None,
        limit: int = 5,
    ) -> tuple[DelegatedAgentTask, ...]:
        return await self._repository.list_for_owner(
            descriptor.owner_person_id,
            TaskListQuery(status, limit, descriptor.session_id),
        )

    async def cancel(
        self,
        descriptor: VoiceSessionDescriptor,
        task: str,
    ) -> CancelOutcome:
        found = await self.get(descriptor, task)
        if found is None:
            return CancelOutcome(None, False)
        outcome = await self._repository.request_cancel(found.id, descriptor.owner_person_id)
        if outcome.cancel_requested:
            await self._wakeup.wake(found.id)
            if (
                outcome.task is not None
                and outcome.task.status.terminal
                and self._completion_notifier is not None
            ):
                try:
                    await self._completion_notifier.wake(outcome.task.owner_person_id)
                except Exception as exc:
                    logger.warning(
                        "Could not signal immediately cancelled delegated task: task=%s error=%s",
                        opaque_ref(str(outcome.task.id)),
                        exception_kind(exc),
                    )
        return outcome
