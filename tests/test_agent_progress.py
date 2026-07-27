from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from cywl_oopz.core.lifecycle import ToolEffect
from cywl_oopz.features.agent.progress import PydanticAiProgressMapper
from cywl_oopz.features.agent.tools.models import ToolDescriptor
from cywl_oopz.features.chat.progress import ProgressKind


class EmptyModel(BaseModel):
    pass


def mapper() -> PydanticAiProgressMapper:
    return PydanticAiProgressMapper(
        (
            ToolDescriptor(
                name="lookup",
                display_name="查询资料",
                description="Look up safe data.",
                input_model=EmptyModel,
                output_model=EmptyModel,
                effect=ToolEffect.READ,
                concurrency_safe=True,
                idempotent=True,
            ),
        )
    )


def test_thinking_parts_and_deltas_are_never_mapped() -> None:
    progress = mapper()

    assert (
        progress.map(
            PartStartEvent(
                index=0,
                part=ThinkingPart("hidden chain of thought"),
            )
        )
        == ()
    )
    assert (
        progress.map(
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=" more hidden reasoning"),
            )
        )
        == ()
    )


def test_text_turn_emits_one_reset_then_visible_deltas() -> None:
    progress = mapper()

    first = progress.map(PartStartEvent(index=0, part=TextPart("你")))
    delta = progress.map(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="好")))

    assert [event.kind for event in first + delta] == [
        ProgressKind.TEXT_RESET,
        ProgressKind.TEXT_DELTA,
        ProgressKind.TEXT_DELTA,
    ]
    assert "".join(event.text for event in first + delta if event.text) == "你好"


def test_tool_events_only_expose_registered_display_identity_and_status() -> None:
    progress = mapper()
    call = ToolCallPart(
        "lookup",
        {"api_key": "must-not-leak"},
        "call-secret",
    )

    started = progress.map(FunctionToolCallEvent(call))[0]
    succeeded = progress.map(
        FunctionToolResultEvent(
            ToolReturnPart(
                "lookup",
                {"ok": True, "data": {"private": "must-not-leak"}},
                "call-secret",
            )
        )
    )[0]
    failed = progress.map(
        FunctionToolResultEvent(
            ToolReturnPart(
                "lookup",
                {"ok": False, "error": "private_error"},
                "call-other",
            )
        )
    )[0]

    assert started.kind is ProgressKind.TOOL_STARTED
    assert started.tool_display_name == "查询资料"
    assert started.tool_detail == ""
    assert succeeded.kind is ProgressKind.TOOL_SUCCEEDED
    assert succeeded.tool_detail == "调用完成"
    assert failed.kind is ProgressKind.TOOL_FAILED
    assert failed.tool_detail == "错误：工具执行失败"
    for event in (started, succeeded, failed):
        assert not hasattr(event, "arguments")
        assert not hasattr(event, "output")
        assert "must-not-leak" not in repr(event)
        assert "private_error" not in repr(event)


def test_web_tool_events_include_bounded_request_result_and_error_details() -> None:
    descriptor = ToolDescriptor(
        name="search_web",
        display_name="搜索公开网页",
        description="Search public pages.",
        input_model=EmptyModel,
        output_model=EmptyModel,
        effect=ToolEffect.READ,
        concurrency_safe=True,
        idempotent=True,
    )
    progress = PydanticAiProgressMapper((descriptor,))

    started = progress.map(
        FunctionToolCallEvent(
            ToolCallPart(
                "search_web",
                {"query": "初音未来 新闻", "time_range": "d"},
                "search-call",
            )
        )
    )[0]
    succeeded = progress.map(
        FunctionToolResultEvent(
            ToolReturnPart(
                "search_web",
                {"ok": True, "data": {"results": [{}, {}, {}]}},
                "search-call",
            )
        )
    )[0]
    failed = progress.map(
        FunctionToolResultEvent(
            ToolReturnPart(
                "search_web",
                {"ok": False, "error": "web_search_timeout"},
                "failed-call",
            )
        )
    )[0]

    assert started.tool_detail == "查询：「初音未来 新闻」 · 时段：d"
    assert succeeded.tool_detail == "找到 3 条结果"
    assert failed.tool_detail == "错误：网页搜索超时"


def test_text_after_tool_call_starts_a_new_draft_generation() -> None:
    progress = mapper()
    progress.map(PartStartEvent(index=0, part=TextPart("旧草稿")))
    progress.map(FunctionToolCallEvent(ToolCallPart("lookup", {}, "call-1")))

    mapped = progress.map(PartStartEvent(index=0, part=TextPart("新回答")))

    assert [event.kind for event in mapped] == [
        ProgressKind.TEXT_RESET,
        ProgressKind.TEXT_DELTA,
    ]
    assert mapped[1].text == "新回答"
