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

    assert rendered == "🎵 **CYWL**\n这是最终回答♪"
    assert "步骤 1" not in rendered


def test_long_final_answer_keeps_head_tail_and_omission_marker() -> None:
    answer = "开" * 1800 + "中" * 1000 + "结" * 800
    rendered = OopzMessageRenderer().render(
        AgentLoopViewState(
            phase=DisplayPhase.SUCCEEDED,
            final_text=answer,
            terminal=True,
        )
    )

    assert rendered.startswith("🎵 **CYWL**\n开")
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
