from __future__ import annotations

import random

import pytest

from cywl_oopz.features.agent.display import (
    AgentLoopViewState,
    DisplayPhase,
    ToolStepStatus,
    ToolStepView,
)
from cywl_oopz.integrations.oopz.message_renderer import (
    OopzMarkupNormalizer,
    OopzMessageRenderer,
    OopzRenderContext,
    OopzTextBudget,
    oopz_units,
)


def step(index: int, status: ToolStepStatus) -> ToolStepView:
    return ToolStepView(
        call_id=f"call-{index}",
        tool_name=f"tool_{index}",
        display_name=f"步骤 {index}",
        status=status,
    )


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        (DisplayPhase.CREATED, "正在准备回答"),
        (DisplayPhase.ACCEPTED, "正在准备回答"),
        (DisplayPhase.THINKING, "正在思考"),
        (DisplayPhase.RETRYING, "正在重新连接"),
        (DisplayPhase.TOOL_RUNNING, "正在处理"),
        (DisplayPhase.DRAFTING, "正在组织回答"),
        (DisplayPhase.CANCELLED, "已取消当前回答"),
    ],
)
def test_renderer_has_a_clear_line_for_each_phase(
    phase: DisplayPhase,
    expected: str,
) -> None:
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(phase=phase, terminal=phase is DisplayPhase.CANCELLED)
    )

    assert expected in rendered
    assert oopz_units(rendered) <= 1950


def test_active_steps_prioritize_running_failure_and_recent_successes() -> None:
    steps = tuple(step(index, ToolStepStatus.SUCCEEDED) for index in range(8)) + (
        step(8, ToolStepStatus.FAILED),
        step(9, ToolStepStatus.RUNNING),
    )
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(phase=DisplayPhase.TOOL_RUNNING, steps=steps)
    )

    assert "步骤 9" in rendered
    assert "步骤 8" in rendered
    assert "步骤 7" in rendered
    assert "步骤 0" not in rendered
    assert "已折叠 5 个" in rendered


def test_drafting_collapses_all_completed_steps_to_one_summary() -> None:
    steps = tuple(step(index, ToolStepStatus.SUCCEEDED) for index in range(20)) + (
        step(20, ToolStepStatus.RUNNING),
    )
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.DRAFTING,
            steps=steps,
            current_draft="回答正在出现",
        )
    )

    assert "已完成 20 个步骤" in rendered
    assert "步骤 0" not in rendered
    assert "步骤 20" in rendered
    assert "回答正在出现" in rendered


def test_drafting_never_exceeds_five_step_lines() -> None:
    steps = tuple(step(index, ToolStepStatus.RUNNING) for index in range(8)) + (
        step(8, ToolStepStatus.SUCCEEDED),
    )
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.DRAFTING,
            steps=steps,
            current_draft="回答",
        )
    )

    progress_lines = [
        line for line in rendered.splitlines() if line.startswith(("⏳", "⚠️", "✅", "… 已折叠"))
    ]
    assert len(progress_lines) == 5


def test_terminal_success_removes_tool_chrome_and_keeps_the_answer() -> None:
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.SUCCEEDED,
            steps=(step(1, ToolStepStatus.SUCCEEDED),),
            final_text="这是最终回答♪",
            terminal=True,
        )
    )

    assert rendered == "🎵 **初音未来**\n这是最终回答♪"
    assert "步骤 1" not in rendered


def test_retrying_model_shows_attempt_countdown_and_safe_reason() -> None:
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.RETRYING,
            provider_retry_count=1,
            retry_attempt=1,
            retry_max_attempts=2,
            retry_delay_seconds=1.25,
            retry_reason="上游服务异常（HTTP 503）",
        ),
        OopzRenderContext(retry_remaining_seconds=0.8, activity_frame=1),
    )

    assert "🔄 **初音未来 正在重新连接..**" in rendered
    assert "↻ 第 1/2 次重试 · 约 0.8s 后继续 · 上游服务异常（HTTP 503）" in rendered


def test_long_final_answer_keeps_head_tail_and_omission_marker() -> None:
    answer = "开" * 1800 + "中" * 1000 + "结" * 800
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.SUCCEEDED,
            final_text=answer,
            terminal=True,
        )
    )

    assert rendered.startswith("🎵 **初音未来**\n开")
    assert "因 OOPZ 长度限制已折叠" in rendered
    assert rendered.endswith("结")
    assert oopz_units(rendered) <= 1950


def test_utf16_budget_counts_emoji_as_two_units() -> None:
    assert oopz_units("a") == 1
    assert oopz_units("你") == 1
    assert oopz_units("😀") == 2
    assert oopz_units("😀" * 975) == 1950

    renderer = OopzMessageRenderer(budget=OopzTextBudget(safe_limit=40))
    rendered = renderer.render(
        AgentLoopViewState(
            phase=DisplayPhase.SUCCEEDED,
            final_text="😀" * 100,
            terminal=True,
        )
    )
    assert oopz_units(rendered) <= 40


@pytest.mark.parametrize("size", [1949, 1950, 1951, 2000, 4000])
def test_every_answer_boundary_is_rendered_within_the_safe_limit(size: int) -> None:
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.SUCCEEDED,
            final_text="答" * size,
            terminal=True,
        )
    )

    assert oopz_units(rendered) <= 1950


def test_normalizer_downgrades_unsupported_markdown() -> None:
    source = """# 标题
- [文档](https://example.com) 与 `变量`

```python
print("hi")
```

| 名称 | 状态 |
| --- | --- |
| CYWL | ok |
"""
    rendered = OopzMarkupNormalizer().normalize(source)

    assert rendered.startswith("**标题**")
    assert "• 文档（https://example.com） 与 「变量」" in rendered
    assert '│ print("hi")' in rendered
    assert "名称 ｜ 状态" in rendered
    assert "---" not in rendered
    assert "```" not in rendered


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("**粗体", "**粗体**"),
        ("~~删除", "~~删除~~"),
        ("*斜体", "*斜体*"),
        ("<u>下划线", "<u>下划线<u>"),
        ("<u>下划线</u>", "<u>下划线<u>"),
        ("***组合***", "***组合***"),
    ],
)
def test_normalizer_repairs_unclosed_supported_markers(
    source: str,
    expected: str,
) -> None:
    assert OopzMarkupNormalizer().normalize(source) == expected


def test_renderer_never_includes_raw_tool_data() -> None:
    state = AgentLoopViewState(
        phase=DisplayPhase.TOOL_RUNNING,
        steps=(
            ToolStepView(
                call_id="secret-call-id",
                tool_name="internal_tool_name",
                display_name="查询状态",
                status=ToolStepStatus.FAILED,
            ),
        ),
    )

    rendered = OopzMessageRenderer().render(state)

    assert "查询状态" in rendered
    assert "secret-call-id" not in rendered
    assert "internal_tool_name" not in rendered


def test_tool_step_shows_request_result_and_readable_error_details() -> None:
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.TOOL_RUNNING,
            steps=(
                ToolStepView(
                    call_id="one",
                    tool_name="search_web",
                    display_name="搜索公开网页",
                    status=ToolStepStatus.SUCCEEDED,
                    subject="「初音未来 新闻」",
                    summary="找到 3 条结果",
                    items=(
                        "https://example.com/one",
                        "https://example.com/two",
                        "https://example.com/three",
                    ),
                    updated_revision=1,
                ),
                ToolStepView(
                    call_id="two",
                    tool_name="read_web_page",
                    display_name="读取网页正文",
                    status=ToolStepStatus.FAILED,
                    subject="example.com",
                    summary="网页响应超时",
                    updated_revision=2,
                ),
            ),
        )
    )

    assert "✅ **搜索公开网页** 「初音未来 新闻」 · 找到 3 条结果" in rendered
    assert "⚠️ **读取网页正文** example.com · 网页响应超时" in rendered
    assert "查询：" not in rendered
    assert "网址：" not in rendered
    assert "错误：" not in rendered


def test_latest_tool_expands_result_urls_without_expanding_older_steps() -> None:
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.THINKING,
            steps=(
                ToolStepView(
                    call_id="old",
                    tool_name="search_web",
                    display_name="搜索公开网页",
                    status=ToolStepStatus.SUCCEEDED,
                    subject="「旧查询」",
                    items=("https://old.example",),
                    updated_revision=1,
                ),
                ToolStepView(
                    call_id="new",
                    tool_name="search_web",
                    display_name="搜索公开网页",
                    status=ToolStepStatus.SUCCEEDED,
                    subject="「初音未来 新闻」",
                    summary="找到 3 条结果",
                    items=(
                        "https://example.com/one",
                        "https://example.com/two",
                        "https://example.com/three",
                    ),
                    updated_revision=2,
                ),
            ),
        )
    )

    assert "https://old.example" not in rendered
    assert "1. https://example.com/one" in rendered
    assert "2. https://example.com/two" in rendered
    assert "3. https://example.com/three" in rendered


def test_running_tool_shows_ephemeral_elapsed_time_without_storing_it_in_state() -> None:
    state = AgentLoopViewState(
        phase=DisplayPhase.TOOL_RUNNING,
        steps=(
            ToolStepView(
                call_id="private-call-id",
                tool_name="read_web_page",
                display_name="读取网页正文",
                status=ToolStepStatus.RUNNING,
                subject="www.baidu.com",
            ),
        ),
    )

    rendered = OopzMessageRenderer().render(
        state,
        OopzRenderContext(
            running_elapsed_seconds=(("private-call-id", 2.36),),
            activity_frame=1,
        ),
    )

    assert "⏳ **读取网页正文** www.baidu.com · 2.4s" in rendered
    assert "private-call-id" not in rendered
    assert not hasattr(state.steps[0], "elapsed_seconds")


def test_tool_region_keeps_its_budget_with_worst_case_parallel_steps() -> None:
    long_subject = "主题" * 40
    long_summary = "摘要" * 50
    long_item = "https://example.com/" + "path/" * 32
    steps = tuple(
        ToolStepView(
            call_id=f"call-{index}",
            tool_name=f"tool_{index}",
            display_name="很长的工具展示名称" * 4,
            status=(ToolStepStatus.RUNNING if index < 3 else ToolStepStatus.SUCCEEDED),
            subject=long_subject,
            summary=long_summary,
            items=(long_item, long_item + "two", long_item + "three"),
            updated_revision=index,
        )
        for index in range(5)
    )
    renderer = OopzMessageRenderer()

    rendered = renderer.render(
        AgentLoopViewState(
            phase=DisplayPhase.TOOL_RUNNING,
            steps=steps,
        )
    )
    tool_region = "\n".join(rendered.splitlines()[1:])

    assert oopz_units(tool_region) <= renderer.max_tool_units
    assert "**很长的工具展示名称" in rendered
    assert "已折叠" in rendered


def test_success_header_shows_compact_agent_run_statistics() -> None:
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.SUCCEEDED,
            final_text="完成啦♪",
            elapsed_seconds=12.34,
            input_tokens=1800,
            output_tokens=345,
            model_requests=3,
            tool_calls=2,
            provider_retry_count=1,
            terminal=True,
        )
    )

    assert rendered == "🎵 **初音未来** · 12.3s · 2 次工具 · 1 次重试 · 2.1k tokens\n完成啦♪"


def test_random_unicode_is_always_bounded_and_balanced() -> None:
    randomizer = random.Random(20260728)
    alphabet = "abc中文😀♪*~<u>`#[]()|\n"
    renderer = OopzMessageRenderer()
    normalizer = OopzMarkupNormalizer()
    for _ in range(100):
        answer = "".join(randomizer.choice(alphabet) for _ in range(4000))
        rendered = renderer.render(
            AgentLoopViewState(
                phase=DisplayPhase.SUCCEEDED,
                final_text=answer,
                terminal=True,
            )
        )
        assert oopz_units(rendered) <= 1950
        plain = normalizer.plain_text(rendered)
        assert "**" not in plain
        assert "~~" not in plain
        assert "<u>" not in plain
