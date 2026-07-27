"""Pure Agent-loop display state and deterministic event reduction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind


class DisplayPhase(StrEnum):
    """User-visible phase of one Agent run."""

    CREATED = "created"
    ACCEPTED = "accepted"
    THINKING = "thinking"
    TOOL_RUNNING = "tool_running"
    DRAFTING = "drafting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolStepStatus(StrEnum):
    """Display-safe state for one model tool call."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ToolStepView:
    """Only presentation-safe tool metadata retained by the reducer."""

    call_id: str
    tool_name: str
    display_name: str
    status: ToolStepStatus
    subject: str = ""
    summary: str = ""
    items: tuple[str, ...] = ()
    preview_lines: tuple[str, ...] = ()
    updated_revision: int = 0


@dataclass(frozen=True, slots=True)
class AgentLoopViewState:
    """Immutable projection of one Agent run."""

    phase: DisplayPhase = DisplayPhase.CREATED
    steps: tuple[ToolStepView, ...] = ()
    current_draft: str = ""
    final_text: str = ""
    terminal_message: str = ""
    elapsed_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_requests: int | None = None
    tool_calls: int | None = None
    completed_step_count: int = 0
    failed_step_count: int = 0
    terminal: bool = False
    revision: int = 0
    seen_event_ids: frozenset[str] = frozenset()


class AgentLoopReducer:
    """Apply progress events without clocks, I/O, SDK types, or side effects."""

    def apply(
        self,
        state: AgentLoopViewState,
        event: ConversationProgressEvent,
    ) -> AgentLoopViewState:
        if state.terminal:
            return state
        if event.event_id and event.event_id in state.seen_event_ids:
            return state

        candidate = self._transition(state, event)
        seen = state.seen_event_ids
        if event.event_id:
            seen = seen | {event.event_id}
        candidate = replace(candidate, seen_event_ids=seen)
        if candidate == replace(state, seen_event_ids=seen):
            return candidate
        return replace(candidate, revision=state.revision + 1)

    def _transition(
        self,
        state: AgentLoopViewState,
        event: ConversationProgressEvent,
    ) -> AgentLoopViewState:
        kind = event.kind
        if kind is ProgressKind.ACCEPTED:
            return replace(state, phase=DisplayPhase.ACCEPTED)
        if kind is ProgressKind.THINKING:
            return replace(
                state,
                phase=(
                    DisplayPhase.TOOL_RUNNING
                    if self._has_running_step(state.steps)
                    else DisplayPhase.THINKING
                ),
            )
        if kind is ProgressKind.TEXT_RESET:
            return replace(state, phase=DisplayPhase.DRAFTING, current_draft="")
        if kind is ProgressKind.TEXT_DELTA:
            return replace(
                state,
                phase=DisplayPhase.DRAFTING,
                current_draft=state.current_draft + event.text,
            )
        if kind is ProgressKind.TOOL_STARTED:
            return self._with_tool(state, event, ToolStepStatus.RUNNING)
        if kind is ProgressKind.TOOL_UPDATED:
            return self._with_tool(state, event, ToolStepStatus.RUNNING)
        if kind is ProgressKind.TOOL_SUCCEEDED:
            return self._with_tool(state, event, ToolStepStatus.SUCCEEDED)
        if kind is ProgressKind.TOOL_FAILED:
            return self._with_tool(state, event, ToolStepStatus.FAILED)
        if kind is ProgressKind.COMPLETED:
            return replace(
                state,
                phase=DisplayPhase.SUCCEEDED,
                current_draft="",
                final_text=event.text,
                elapsed_seconds=event.elapsed_seconds,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                model_requests=event.model_requests,
                tool_calls=event.tool_calls,
                terminal=True,
            )
        if kind is ProgressKind.FAILED:
            return replace(
                state,
                phase=DisplayPhase.FAILED,
                current_draft="",
                terminal_message=event.text,
                terminal=True,
            )
        if kind is ProgressKind.CANCELLED:
            return replace(
                state,
                phase=DisplayPhase.CANCELLED,
                current_draft="",
                terminal=True,
            )
        return state

    def _with_tool(
        self,
        state: AgentLoopViewState,
        event: ConversationProgressEvent,
        status: ToolStepStatus,
    ) -> AgentLoopViewState:
        steps = list(state.steps)
        for index, current in enumerate(steps):
            if current.call_id == event.call_id:
                step = ToolStepView(
                    call_id=event.call_id,
                    tool_name=event.tool_name,
                    display_name=event.tool_display_name,
                    status=status,
                    subject=event.tool_subject or current.subject,
                    summary=event.tool_summary or current.summary,
                    items=(
                        () if status is ToolStepStatus.FAILED else event.tool_items or current.items
                    ),
                    preview_lines=(
                        ()
                        if status is ToolStepStatus.FAILED
                        else event.tool_preview_lines or current.preview_lines
                    ),
                    updated_revision=current.updated_revision,
                )
                if current == step:
                    return state
                step = replace(step, updated_revision=state.revision + 1)
                steps[index] = step
                break
        else:
            steps.append(
                ToolStepView(
                    call_id=event.call_id,
                    tool_name=event.tool_name,
                    display_name=event.tool_display_name,
                    status=status,
                    subject=event.tool_subject,
                    summary=event.tool_summary,
                    items=event.tool_items,
                    preview_lines=event.tool_preview_lines,
                    updated_revision=state.revision + 1,
                )
            )
        normalized = tuple(steps)
        completed = sum(item.status is ToolStepStatus.SUCCEEDED for item in normalized)
        failed = sum(item.status is ToolStepStatus.FAILED for item in normalized)
        phase = (
            DisplayPhase.TOOL_RUNNING
            if self._has_running_step(normalized)
            else DisplayPhase.THINKING
        )
        return replace(
            state,
            phase=phase,
            steps=normalized,
            current_draft="",
            completed_step_count=completed,
            failed_step_count=failed,
        )

    @staticmethod
    def _has_running_step(steps: tuple[ToolStepView, ...]) -> bool:
        return any(step.status is ToolStepStatus.RUNNING for step in steps)
