"""Project-owned interfaces for Agent tools and their side effects."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pydantic import BaseModel

from .models import (
    ToolCall,
    ToolDescriptor,
    ToolExecution,
    ToolExecutionClaim,
    ToolExecutionContext,
    ToolExecutionResult,
    ToolExecutionStatus,
)


class AgentTool(Protocol):
    """One explicitly constructed application tool."""

    @property
    def descriptor(self) -> ToolDescriptor:
        """Return stable schema and execution policy."""

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        """Execute with trusted context and validated arguments."""


class ToolExecutionRepository(Protocol):
    """Persistence boundary for idempotent tool execution records."""

    async def claim(self, execution: ToolExecution) -> ToolExecutionClaim:
        """Atomically insert or return the existing run/tool-call record."""

    async def finish(
        self,
        run_id: UUID,
        call_id: str,
        status: ToolExecutionStatus,
        *,
        output: dict[str, object] | None,
        error_code: str,
    ) -> ToolExecution:
        """Persist one terminal state and return the stored record."""


class AgentToolRuntime(Protocol):
    """Narrow boundary consumed by the Pydantic AI adapter."""

    def descriptors(self, names: tuple[str, ...]) -> tuple[ToolDescriptor, ...]:
        """Resolve only an already-authorized set of names."""

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        """Execute one tool call with policy, timeout, and persistence."""
