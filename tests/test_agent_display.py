from __future__ import annotations

import pytest

from cywl_oopz.features.agent.display import (
    AgentLoopReducer,
    AgentLoopViewState,
    DisplayPhase,
    ToolStepStatus,
)
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind


def event(kind: ProgressKind, **values: str) -> ConversationProgressEvent:
    return ConversationProgressEvent(kind, **values)


def tool_event(
    kind: ProgressKind,
    call_id: str,
    *,
    name: str = "search_music_catalog",
    display_name: str = "搜索歌曲",
    event_id: str = "",
    detail: str = "",
) -> ConversationProgressEvent:
    return event(
        kind,
        call_id=call_id,
        tool_name=name,
        tool_display_name=display_name,
        tool_detail=detail,
        event_id=event_id,
    )


def reduce(
    *events: ConversationProgressEvent,
    initial: AgentLoopViewState | None = None,
) -> AgentLoopViewState:
    reducer = AgentLoopReducer()
    state = initial or AgentLoopViewState()
    for item in events:
        state = reducer.apply(state, item)
    return state


def test_simple_text_run_reaches_an_immutable_terminal_state() -> None:
    state = reduce(
        event(ProgressKind.ACCEPTED),
        event(ProgressKind.THINKING),
        event(ProgressKind.TEXT_RESET),
        event(ProgressKind.TEXT_DELTA, text="你好"),
        event(ProgressKind.TEXT_DELTA, text="呀♪"),
        event(ProgressKind.COMPLETED, text="你好呀♪"),
    )

    assert state.phase is DisplayPhase.SUCCEEDED
    assert state.final_text == "你好呀♪"
    assert state.current_draft == ""
    assert state.terminal is True
    assert state.revision == 6

    assert (
        AgentLoopReducer().apply(
            state,
            event(ProgressKind.TEXT_DELTA, text="late"),
        )
        is state
    )


def test_tool_run_clears_temporary_draft_and_tracks_safe_metadata_only() -> None:
    state = reduce(
        event(ProgressKind.TEXT_RESET),
        event(ProgressKind.TEXT_DELTA, text="临时回答"),
        tool_event(ProgressKind.TOOL_STARTED, "call-1", detail="查询：「Tell Your World」"),
        tool_event(ProgressKind.TOOL_SUCCEEDED, "call-1", detail="找到 3 首歌曲"),
        event(ProgressKind.THINKING),
    )

    assert state.phase is DisplayPhase.THINKING
    assert state.current_draft == ""
    assert state.completed_step_count == 1
    assert state.failed_step_count == 0
    assert state.steps[0].status is ToolStepStatus.SUCCEEDED
    assert state.steps[0].request_detail == "查询：「Tell Your World」"
    assert state.steps[0].result_detail == "找到 3 首歌曲"
    assert not hasattr(state.steps[0], "arguments")
    assert not hasattr(state.steps[0], "output")


def test_failure_can_be_recovered_by_a_later_status_for_the_same_call() -> None:
    failed = reduce(
        tool_event(ProgressKind.TOOL_STARTED, "call-1"),
        tool_event(ProgressKind.TOOL_FAILED, "call-1"),
    )
    assert failed.failed_step_count == 1

    recovered = reduce(
        tool_event(ProgressKind.TOOL_SUCCEEDED, "call-1"),
        initial=failed,
    )
    assert recovered.failed_step_count == 0
    assert recovered.completed_step_count == 1
    assert len(recovered.steps) == 1


def test_parallel_tools_keep_insertion_order_when_they_finish_out_of_order() -> None:
    state = reduce(
        tool_event(ProgressKind.TOOL_STARTED, "one", display_name="第一步"),
        tool_event(ProgressKind.TOOL_STARTED, "two", display_name="第二步"),
        tool_event(ProgressKind.TOOL_STARTED, "three", display_name="第三步"),
        tool_event(ProgressKind.TOOL_SUCCEEDED, "two", display_name="第二步"),
        tool_event(ProgressKind.TOOL_FAILED, "one", display_name="第一步"),
        tool_event(ProgressKind.TOOL_SUCCEEDED, "three", display_name="第三步"),
    )

    assert [step.call_id for step in state.steps] == ["one", "two", "three"]
    assert [step.status for step in state.steps] == [
        ToolStepStatus.FAILED,
        ToolStepStatus.SUCCEEDED,
        ToolStepStatus.SUCCEEDED,
    ]
    assert state.completed_step_count == 2
    assert state.failed_step_count == 1


def test_new_model_turn_resets_the_previous_draft() -> None:
    state = reduce(
        event(ProgressKind.TEXT_RESET),
        event(ProgressKind.TEXT_DELTA, text="旧回答"),
        event(ProgressKind.TEXT_RESET),
        event(ProgressKind.TEXT_DELTA, text="新回答"),
    )

    assert state.phase is DisplayPhase.DRAFTING
    assert state.current_draft == "新回答"


def test_duplicate_event_id_and_duplicate_tool_status_are_idempotent() -> None:
    reducer = AgentLoopReducer()
    state = reducer.apply(
        AgentLoopViewState(),
        tool_event(ProgressKind.TOOL_STARTED, "one", event_id="event-1"),
    )
    duplicate_id = reducer.apply(
        state,
        tool_event(ProgressKind.TOOL_FAILED, "one", event_id="event-1"),
    )
    duplicate_status = reducer.apply(
        state,
        tool_event(ProgressKind.TOOL_STARTED, "one"),
    )

    assert duplicate_id is state
    assert duplicate_status.revision == state.revision
    assert duplicate_status.steps == state.steps


@pytest.mark.parametrize(
    ("kind", "phase", "message"),
    [
        (ProgressKind.FAILED, DisplayPhase.FAILED, "服务暂时不可用"),
        (ProgressKind.CANCELLED, DisplayPhase.CANCELLED, ""),
    ],
)
def test_non_success_terminal_states_are_explicit(
    kind: ProgressKind,
    phase: DisplayPhase,
    message: str,
) -> None:
    values = {"text": message} if message else {}
    state = reduce(event(kind, **values))

    assert state.phase is phase
    assert state.terminal is True
    assert state.terminal_message == message


def test_progress_event_rejects_raw_or_incomplete_shapes() -> None:
    with pytest.raises(ValueError, match="call_id"):
        event(ProgressKind.TOOL_STARTED)
    with pytest.raises(ValueError, match="display name"):
        event(
            ProgressKind.TOOL_STARTED,
            call_id="call",
            tool_name="tool",
        )
    with pytest.raises(ValueError, match="short line"):
        tool_event(ProgressKind.TOOL_STARTED, "call", display_name="bad\nname")
    with pytest.raises(ValueError, match="detail"):
        tool_event(ProgressKind.TOOL_STARTED, "call", detail="bad\ndetail")
    with pytest.raises(ValueError, match="requires text"):
        event(ProgressKind.COMPLETED)
    with pytest.raises(TypeError):
        ConversationProgressEvent(ProgressKind.TOOL_STARTED, arguments={"secret": True})  # type: ignore[call-arg]


def test_completed_event_carries_terminal_run_statistics() -> None:
    state = reduce(
        event(
            ProgressKind.COMPLETED,
            text="完成",
            elapsed_seconds=12.34,
            input_tokens=1200,
            output_tokens=345,
            model_requests=3,
            tool_calls=2,
        )
    )

    assert state.elapsed_seconds == 12.34
    assert state.input_tokens == 1200
    assert state.output_tokens == 345
    assert state.model_requests == 3
    assert state.tool_calls == 2
