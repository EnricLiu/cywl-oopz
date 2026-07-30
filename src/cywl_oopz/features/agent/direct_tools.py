"""Direct Agent tool execution for explicit chat-command debugging."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from cywl_oopz.features.chat.models import ChatInvocation, ConversationKey
from cywl_oopz.settings import AgentSettings

from .models import AgentIdentity, AgentRunLimits, ModelCapability
from .selection import ProviderSelectionService
from .tools.executor import ToolExecutor
from .tools.models import (
    ToolCall,
    ToolDescriptor,
    ToolExecution,
    ToolExecutionClaim,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from .tools.policy import ToolAvailabilityService, ToolPolicy
from .tools.registry import ToolRegistry


class _SingleCallExecutionRepository:
    """Keep one debug call's executor state without requiring a persisted Agent run."""

    def __init__(self) -> None:
        self._execution: ToolExecution | None = None

    async def claim(self, execution: ToolExecution) -> ToolExecutionClaim:
        if self._execution is None:
            self._execution = execution
            return ToolExecutionClaim(execution, created=True)
        return ToolExecutionClaim(self._execution, created=False)

    async def finish(
        self,
        run_id: UUID,
        call_id: str,
        status: ToolExecutionStatus,
        *,
        output: dict[str, object] | None,
        error_code: str,
    ) -> ToolExecution:
        execution = self._execution
        if execution is None or execution.run_id != run_id or execution.call_id != call_id:
            raise RuntimeError("Direct tool execution was not claimed")
        execution = replace(
            execution,
            status=status,
            output_payload=output,
            error_code=error_code,
            finished_at=datetime.now(UTC),
        )
        self._execution = execution
        return execution


class DirectToolService:
    """Describe or execute one curated tool without entering the Agent loop."""

    _REQUIRED_CAPABILITIES = frozenset({ModelCapability.TOOL_CALLING})

    def __init__(
        self,
        settings: AgentSettings,
        registry: ToolRegistry,
        availability: ToolAvailabilityService,
        selection: ProviderSelectionService,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._availability = availability
        self._selection = selection
        self._policy = ToolPolicy()

    def describe(self, tool_name: str) -> dict[str, object] | None:
        """Return stable metadata and validation schemas for one registered tool."""
        tool = self._registry.get(tool_name)
        if tool is None:
            return None
        descriptor = tool.descriptor
        return self._descriptor_payload(descriptor)

    async def execute(
        self,
        key: ConversationKey,
        invocation: ChatInvocation,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ToolExecutionResult:
        """Execute with the same availability, policy, validation, and timeout rules."""
        identity = AgentIdentity(
            key.person_id,
            key,
            source_message_id=invocation.source_message_id,
            transport_channel_id=invocation.transport_channel_id,
            mentioned_person_ids=invocation.mentioned_person_ids,
        )
        selection = await self._selection.resolve(
            key,
            required_capabilities=self._REQUIRED_CAPABILITIES,
        )
        enabled_tools = await self._availability.names(identity, selection.model)
        run_id = uuid4()
        context = self._execution_context(run_id, identity, enabled_tools)
        executor = ToolExecutor(
            self._registry,
            self._policy,
            _SingleCallExecutionRepository(),
        )
        return await executor.execute(
            ToolCall(
                call_id=f"direct-{uuid4().hex}",
                name=tool_name,
                arguments=arguments,
            ),
            context,
        )

    def _execution_context(
        self,
        run_id: UUID,
        identity: AgentIdentity,
        enabled_tools: tuple[str, ...],
    ) -> ToolExecutionContext:
        return ToolExecutionContext(
            run_id=run_id,
            identity=identity,
            limits=AgentRunLimits(
                timeout_seconds=self._settings.timeout_seconds,
                max_model_requests=self._settings.max_model_requests,
                max_tool_calls=self._settings.max_tool_calls,
                max_total_tokens=self._settings.max_total_tokens,
                max_parallel_tools=self._settings.max_parallel_tools,
            ),
            enabled_tools=enabled_tools,
        )

    @staticmethod
    def _descriptor_payload(descriptor: ToolDescriptor) -> dict[str, object]:
        return {
            "id": descriptor.name,
            "display_name": descriptor.display_name,
            "description": descriptor.description,
            "effect": descriptor.effect.value,
            "version": descriptor.version,
            "timeout_seconds": descriptor.timeout_seconds,
            "max_retries": descriptor.max_retries,
            "max_output_characters": descriptor.max_output_characters,
            "concurrency_safe": descriptor.concurrency_safe,
            "sequential": descriptor.sequential,
            "idempotent": descriptor.idempotent,
            "replay_in_history": descriptor.replay_in_history,
            "input_schema": descriptor.input_model.model_json_schema(),
            "output_schema": descriptor.output_model.model_json_schema(),
        }
