"""Policy-bounded, idempotent Agent tool execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from cywl_oopz.core.observability import exception_kind

from .models import (
    ToolCall,
    ToolDescriptor,
    ToolEffect,
    ToolExecution,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from .policy import ToolPolicy
from .ports import AgentTool, ToolExecutionRepository
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Validate, authorize, bound, persist, and execute registered tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy,
        executions: ToolExecutionRepository,
    ) -> None:
        self._registry = registry
        self._policy = policy
        self._executions = executions

    def descriptors(self, names: tuple[str, ...]) -> tuple[ToolDescriptor, ...]:
        """Resolve only names previously selected by ToolAvailabilityService."""
        return self._registry.descriptors(names)

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        """Execute one call and convert all tool failures to safe model data."""
        try:
            tool = self._registry.get(call.name)
            descriptor = tool.descriptor if tool is not None else None
        except Exception as exc:
            logger.exception(
                "Agent tool registry failed: run=%s call=%s tool=%s phase=tool_validate error=%s",
                context.run_id,
                call.call_id,
                call.name,
                exception_kind(exc),
            )
            return ToolExecutionResult(
                call.call_id,
                call.name,
                ToolExecutionStatus.FAILED,
                error_code="tool_failed",
            )
        if tool is None:
            logger.warning(
                "Rejected unregistered Agent tool: run=%s tool=%s",
                context.run_id,
                call.name,
            )
            return ToolExecutionResult(
                call.call_id,
                call.name,
                ToolExecutionStatus.DENIED,
                error_code="tool_not_registered",
            )

        assert descriptor is not None
        try:
            arguments = descriptor.input_model.model_validate(dict(call.arguments))
        except ValidationError:
            logger.warning(
                "Rejected Agent tool arguments: run=%s call=%s tool=%s phase=tool_validate",
                context.run_id,
                call.call_id,
                descriptor.name,
            )
            return ToolExecutionResult(
                call.call_id,
                call.name,
                ToolExecutionStatus.FAILED,
                error_code="invalid_arguments",
            )
        except Exception as exc:
            logger.exception(
                "Agent tool argument validation crashed: run=%s call=%s tool=%s "
                "phase=tool_validate error=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                exception_kind(exc),
            )
            return ToolExecutionResult(
                call.call_id,
                call.name,
                ToolExecutionStatus.FAILED,
                error_code="tool_failed",
            )

        try:
            normalized_call = ToolCall(
                call.call_id,
                call.name,
                arguments.model_dump(mode="json"),
            )
            pending_execution = ToolExecution(
                id=uuid4(),
                run_id=context.run_id,
                call_id=normalized_call.call_id,
                tool_name=descriptor.name,
                tool_version=descriptor.version,
                effect=descriptor.effect,
                status=ToolExecutionStatus.STARTED,
                idempotency_key=self._idempotency_key(
                    context,
                    normalized_call,
                    descriptor.effect,
                ),
                input_payload=self._persisted_input(normalized_call, descriptor),
                output_payload=None,
                error_code="",
                started_at=datetime.now(UTC),
            )
        except Exception as exc:
            logger.exception(
                "Failed to normalize Agent tool state: run=%s call=%s tool=%s "
                "phase=tool_claim error=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                exception_kind(exc),
            )
            return ToolExecutionResult(
                call.call_id,
                call.name,
                ToolExecutionStatus.FAILED,
                error_code="tool_state_unavailable",
            )

        try:
            claim = await self._executions.claim(pending_execution)
            if not claim.created:
                logger.info(
                    "Reused existing Agent tool execution: run=%s call=%s tool=%s status=%s",
                    context.run_id,
                    call.call_id,
                    call.name,
                    claim.execution.status.value,
                )
                return self._from_execution(claim.execution)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Agent tool state claim failed: run=%s call=%s tool=%s phase=tool_claim error=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                exception_kind(exc),
            )
            return ToolExecutionResult(
                call.call_id,
                call.name,
                ToolExecutionStatus.FAILED,
                error_code="tool_state_unavailable",
            )

        try:
            return await self._execute_claimed(
                normalized_call,
                context,
                descriptor,
                tool,
                arguments,
            )
        except asyncio.CancelledError:
            logger.info(
                "Agent tool execution cancelled: run=%s call=%s tool=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
            )
            await self._finish_after_cancellation(context, normalized_call)
            raise
        except Exception as exc:
            logger.exception(
                "Agent tool boundary crashed: run=%s call=%s tool=%s phase=tool_execute error=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                exception_kind(exc),
            )
            return await self._finish_safely(
                context,
                normalized_call,
                ToolExecutionStatus.FAILED,
                error_code="tool_failed",
            )

    async def _execute_claimed(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        descriptor: ToolDescriptor,
        tool: AgentTool,
        arguments: BaseModel,
    ) -> ToolExecutionResult:
        """Execute one validated, durably claimed call."""
        try:
            error_code = await self._policy.denial_reason(context, descriptor)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Agent tool policy failed: run=%s call=%s tool=%s phase=tool_policy error=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                exception_kind(exc),
            )
            return await self._finish_safely(
                context,
                call,
                ToolExecutionStatus.FAILED,
                error_code="tool_failed",
            )
        if error_code:
            logger.warning(
                "Denied Agent tool execution: run=%s call=%s tool=%s reason=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                error_code,
            )
            return await self._finish_safely(
                context,
                call,
                ToolExecutionStatus.DENIED,
                error_code=error_code,
            )

        try:
            logger.info(
                "Agent tool execution started: run=%s call=%s tool=%s timeout_seconds=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                descriptor.timeout_seconds,
            )
            async with asyncio.timeout(descriptor.timeout_seconds):
                raw_output = await tool.execute(context, arguments)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            logger.warning(
                "Agent tool execution timed out: run=%s call=%s tool=%s error=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                exception_kind(exc),
            )
            return await self._finish_safely(
                context,
                call,
                ToolExecutionStatus.FAILED,
                error_code="tool_timeout",
            )
        except ToolExecutionError as exc:
            logger.warning(
                "Agent tool execution failed: run=%s call=%s tool=%s reason=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                exc.error_code,
            )
            return await self._finish_safely(
                context,
                call,
                ToolExecutionStatus.FAILED,
                error_code=exc.error_code,
            )
        except Exception as exc:
            logger.exception(
                "Agent tool execution crashed: run=%s call=%s tool=%s phase=tool_execute error=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                exception_kind(exc),
            )
            return await self._finish_safely(
                context,
                call,
                ToolExecutionStatus.FAILED,
                error_code="tool_failed",
            )

        try:
            output_model = descriptor.output_model.model_validate(raw_output)
            output = output_model.model_dump(mode="json")
            bounded = self._bounded_output(output, descriptor.max_output_characters)
        except ValidationError:
            logger.warning(
                "Agent tool returned invalid output: run=%s call=%s tool=%s phase=tool_output",
                context.run_id,
                call.call_id,
                descriptor.name,
            )
            return await self._finish_safely(
                context,
                call,
                ToolExecutionStatus.FAILED,
                error_code="invalid_tool_output",
            )
        except Exception as exc:
            logger.exception(
                "Agent tool output normalization failed: run=%s call=%s tool=%s "
                "phase=tool_output error=%s",
                context.run_id,
                call.call_id,
                descriptor.name,
                exception_kind(exc),
            )
            return await self._finish_safely(
                context,
                call,
                ToolExecutionStatus.FAILED,
                error_code="invalid_tool_output",
            )
        result = await self._finish_safely(
            context,
            call,
            ToolExecutionStatus.SUCCEEDED,
            output=bounded,
        )
        logger.info(
            "Agent tool execution completed: run=%s call=%s tool=%s output_truncated=%s",
            context.run_id,
            call.call_id,
            descriptor.name,
            bounded.get("truncated") is True,
        )
        return result

    async def _finish_safely(
        self,
        context: ToolExecutionContext,
        call: ToolCall,
        status: ToolExecutionStatus,
        *,
        output: dict[str, object] | None = None,
        error_code: str = "",
    ) -> ToolExecutionResult:
        try:
            return await self._finish(
                context,
                call,
                status,
                output=output,
                error_code=error_code,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Agent tool state finish failed: run=%s call=%s tool=%s "
                "phase=tool_finish primary_status=%s primary_error=%s error=%s",
                context.run_id,
                call.call_id,
                call.name,
                status.value,
                error_code or "none",
                exception_kind(exc),
            )
            return ToolExecutionResult(
                call.call_id,
                call.name,
                ToolExecutionStatus.FAILED,
                error_code="tool_state_unavailable",
            )

    async def _finish_after_cancellation(
        self,
        context: ToolExecutionContext,
        call: ToolCall,
    ) -> None:
        try:
            await asyncio.shield(
                self._finish(
                    context,
                    call,
                    ToolExecutionStatus.CANCELLED,
                    error_code="cancelled",
                )
            )
        except asyncio.CancelledError:
            logger.warning(
                "Agent tool cancellation cleanup interrupted: run=%s call=%s tool=%s "
                "phase=tool_finish",
                context.run_id,
                call.call_id,
                call.name,
            )
        except Exception as exc:
            logger.exception(
                "Agent tool cancellation cleanup failed: run=%s call=%s tool=%s "
                "phase=tool_finish primary_status=cancelled error=%s",
                context.run_id,
                call.call_id,
                call.name,
                exception_kind(exc),
            )

    async def _finish(
        self,
        context: ToolExecutionContext,
        call: ToolCall,
        status: ToolExecutionStatus,
        *,
        output: dict[str, object] | None = None,
        error_code: str = "",
    ) -> ToolExecutionResult:
        execution = await self._executions.finish(
            context.run_id,
            call.call_id,
            status,
            output=output,
            error_code=error_code,
        )
        return self._from_execution(execution)

    @staticmethod
    def _idempotency_key(
        context: ToolExecutionContext,
        call: ToolCall,
        effect: ToolEffect,
    ) -> str:
        if effect is ToolEffect.READ:
            return f"{context.run_id}:{call.call_id}"
        canonical_arguments = json.dumps(
            dict(call.arguments),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical_arguments.encode()).hexdigest()
        return f"{context.run_id}:{call.name}:{digest}"

    @staticmethod
    def _persisted_input(
        call: ToolCall,
        descriptor: ToolDescriptor,
    ) -> dict[str, object]:
        if descriptor.persist_input_payload:
            return dict(call.arguments)
        serialized = json.dumps(
            dict(call.arguments),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "redacted": True,
            "argument_characters": len(serialized),
        }

    @staticmethod
    def _bounded_output(
        output: dict[str, object],
        max_characters: int,
    ) -> dict[str, object]:
        serialized = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) <= max_characters:
            return output
        candidate: dict[str, object] = {
            "truncated": True,
            "preview": serialized[: max_characters // 2],
        }
        while (
            len(
                json.dumps(
                    candidate,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            > max_characters
        ):
            preview = str(candidate["preview"])
            candidate["preview"] = preview[: max(0, len(preview) // 2)]
        return candidate

    @staticmethod
    def _from_execution(execution: ToolExecution) -> ToolExecutionResult:
        if execution.status is ToolExecutionStatus.STARTED:
            return ToolExecutionResult(
                execution.call_id,
                execution.tool_name,
                ToolExecutionStatus.DENIED,
                error_code="duplicate_tool_call_in_progress",
            )
        return ToolExecutionResult(
            execution.call_id,
            execution.tool_name,
            execution.status,
            execution.output_payload or {},
            execution.error_code,
        )
