"""Five small realtime tools for controlling durable background Agent tasks."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.agent.delegation.models import (
    DelegatedAgentTask,
    DelegatedResultStyle,
    DelegatedTaskStatus,
)
from cywl_oopz.features.agent.delegation.service import VoiceDelegatedTaskService

from .models import VoiceSessionDescriptor

logger = logging.getLogger(__name__)


class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DelegateAgentTaskArguments(_ToolArguments):
    objective: str = Field(min_length=1, max_length=2000)
    result_style: Literal["brief", "detailed"] = "brief"


class TaskReferenceArguments(_ToolArguments):
    task: str = Field(min_length=2, max_length=64)


class ListAgentTasksArguments(_ToolArguments):
    status: (
        Literal[
            "queued",
            "running",
            "waiting_retry",
            "cancel_requested",
            "succeeded",
            "failed",
            "cancelled",
            "interrupted",
        ]
        | None
    ) = None
    limit: int = Field(default=5, ge=1, le=5)


class VoiceTaskControlTools:
    """Validate Provider calls and return bounded JSON-compatible envelopes."""

    _ARGUMENT_MODELS: dict[str, type[_ToolArguments]] = {
        "delegate_agent_task": DelegateAgentTaskArguments,
        "get_agent_task": TaskReferenceArguments,
        "list_agent_tasks": ListAgentTasksArguments,
        "read_agent_task_result": TaskReferenceArguments,
        "cancel_agent_task": TaskReferenceArguments,
    }
    _DESCRIPTIONS = {
        "delegate_agent_task": (
            "把需要搜索、读网页或频道策略允许的慢速修改目标交给后台 Agent，任务入队后立即返回。"
        ),
        "get_agent_task": "查询当前用户的一个后台任务状态和简短进度。",
        "list_agent_tasks": "列出当前语音会话最近的后台任务。",
        "read_agent_task_result": "读取一个已完成后台任务的结果。",
        "cancel_agent_task": "请求停止一个后台任务；已经发生的操作不会回滚。",
    }

    def __init__(self, service: VoiceDelegatedTaskService) -> None:
        self._service = service

    @classmethod
    def schemas(cls) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "type": "function",
                "name": name,
                "description": cls._DESCRIPTIONS[name],
                "parameters": model.model_json_schema(),
            }
            for name, model in cls._ARGUMENT_MODELS.items()
        )

    async def execute(
        self,
        descriptor: VoiceSessionDescriptor,
        call_id: str,
        name: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        model = self._ARGUMENT_MODELS.get(name)
        if model is None:
            return {"ok": False, "code": "tool_not_allowed"}
        try:
            parsed = model.model_validate(dict(arguments))
            if name == "delegate_agent_task":
                assert isinstance(parsed, DelegateAgentTaskArguments)
                task = await self._service.delegate(
                    descriptor,
                    call_id,
                    parsed.objective,
                    DelegatedResultStyle(parsed.result_style),
                )
                return {
                    "ok": True,
                    "accepted": True,
                    "task": task.alias,
                    "status": task.status.value,
                    "message": f"后台任务 {task.alias} 已排队，可以继续聊。",
                }
            if name == "list_agent_tasks":
                assert isinstance(parsed, ListAgentTasksArguments)
                tasks = await self._service.list(
                    descriptor,
                    status=DelegatedTaskStatus(parsed.status) if parsed.status else None,
                    limit=parsed.limit,
                )
                return {
                    "ok": True,
                    "tasks": [_task_summary(task) for task in tasks],
                }
            assert isinstance(parsed, TaskReferenceArguments)
            if name == "cancel_agent_task":
                outcome = await self._service.cancel(descriptor, parsed.task)
                if outcome.task is None:
                    return {"ok": False, "code": "task_not_found"}
                return {
                    "ok": True,
                    "task": outcome.task.alias,
                    "status": outcome.task.status.value,
                    "cancel_requested": outcome.cancel_requested,
                    "already_terminal": outcome.already_terminal,
                }
            task = await self._service.get(descriptor, parsed.task)
            if task is None:
                return {"ok": False, "code": "task_not_found"}
            if name == "get_agent_task":
                return {"ok": True, **_task_summary(task)}
            return {"ok": True, **_task_result(task)}
        except (ValidationError, ValueError):
            return {"ok": False, "code": "invalid_arguments"}
        except DatabaseError as exc:
            logger.warning(
                "Realtime task control database failure: session=%s tool=%s error=%s",
                opaque_ref(str(descriptor.session_id)),
                name,
                exception_kind(exc),
            )
            return {"ok": False, "code": "temporarily_unavailable"}
        except Exception as exc:
            logger.exception(
                "Realtime task control failed: session=%s tool=%s error=%s",
                opaque_ref(str(descriptor.session_id)),
                name,
                exception_kind(exc),
            )
            return {"ok": False, "code": "internal_error"}


def _task_summary(task: DelegatedAgentTask) -> dict[str, object]:
    now = datetime.now(UTC)
    started = task.created_at or now
    ended = task.finished_at or now
    elapsed_seconds = max(0, int((ended - started).total_seconds()))
    payload: dict[str, object] = {
        "task": task.alias,
        "status": task.status.value,
        "elapsed_seconds": elapsed_seconds,
    }
    if task.progress_stage:
        payload["stage"] = task.progress_stage
    if task.progress_summary:
        payload["progress"] = task.progress_summary[:240]
    if task.status.terminal and task.result_summary:
        payload["summary"] = task.result_summary[:500]
    if task.status in {DelegatedTaskStatus.FAILED, DelegatedTaskStatus.INTERRUPTED}:
        payload["error"] = task.error_code or "task_failed"
    return payload


def _task_result(task: DelegatedAgentTask) -> dict[str, object]:
    payload = _task_summary(task)
    if not task.status.terminal:
        payload["available"] = False
        return payload
    payload["available"] = task.status is DelegatedTaskStatus.SUCCEEDED
    if task.status is DelegatedTaskStatus.SUCCEEDED:
        payload["result"] = (task.result_text or task.result_summary)[:3000]
        payload["truncated"] = len(task.result_text or task.result_summary) > 3000
    elif task.error_message:
        payload["message"] = task.error_message[:500]
    return payload
