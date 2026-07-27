"""Render Agent-loop state into bounded OOPZ-compatible text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cywl_oopz.features.agent.display import (
    AgentLoopViewState,
    DisplayPhase,
    ToolStepStatus,
    ToolStepView,
)

_FENCE = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)
_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$")
_LINK = re.compile(r"\[([^\]\n]+)]\(([^)\n]+)\)")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BULLET = re.compile(r"(?m)^\s*[-+]\s+")
_UNSUPPORTED_HTML = re.compile(r"</?(?!u(?:>|$))[A-Za-z][^>\n]*>")
_MARKERS = ("<u>", "**", "~~", "*")


def oopz_units(text: str) -> int:
    """Count UTF-16 code units conservatively for OOPZ clients."""
    return len(text.encode("utf-16-le")) // 2


def _take_prefix(text: str, units: int) -> str:
    if units <= 0:
        return ""
    used = 0
    parts: list[str] = []
    for character in text:
        size = oopz_units(character)
        if used + size > units:
            break
        parts.append(character)
        used += size
    return "".join(parts)


def _take_suffix(text: str, units: int) -> str:
    if units <= 0:
        return ""
    used = 0
    parts: list[str] = []
    for character in reversed(text):
        size = oopz_units(character)
        if used + size > units:
            break
        parts.append(character)
        used += size
    return "".join(reversed(parts))


class OopzMarkupNormalizer:
    """Downgrade common Markdown and repair OOPZ's four style markers."""

    def normalize(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("</u>", "<u>")
        normalized = _FENCE.sub(self._code_block, normalized)
        normalized = _HEADING.sub(lambda match: f"**{match.group(1)}**", normalized)
        normalized = _LINK.sub(
            lambda match: f"{match.group(1)}（{match.group(2)}）",
            normalized,
        )
        normalized = _INLINE_CODE.sub(lambda match: f"「{match.group(1)}」", normalized)
        normalized = _BULLET.sub("• ", normalized)
        normalized = self._normalize_tables(normalized)
        normalized = _UNSUPPORTED_HTML.sub("", normalized)
        normalized = self._balance_supported_markers(normalized)
        return re.sub(r"\n{3,}", "\n\n", normalized).strip()

    def plain_text(self, text: str) -> str:
        """Return normalized visible text without any style delimiters."""
        normalized = self.normalize(text)
        for marker in _MARKERS:
            normalized = normalized.replace(marker, "")
        return normalized

    @staticmethod
    def _code_block(match: re.Match[str]) -> str:
        content = match.group(1).strip("\n")
        return "\n".join(f"│ {line}" if line else "│" for line in content.splitlines())

    @staticmethod
    def _normalize_tables(text: str) -> str:
        lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if "|" not in stripped:
                lines.append(line)
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
                continue
            lines.append(" ｜ ".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _balance_supported_markers(text: str) -> str:
        result: list[str] = []
        stack: list[str] = []
        index = 0
        while index < len(text):
            marker = next(
                (candidate for candidate in _MARKERS if text.startswith(candidate, index)),
                None,
            )
            if marker is None:
                result.append(text[index])
                index += 1
                continue
            if stack and stack[-1] == marker:
                stack.pop()
                result.append(marker)
            elif marker in stack:
                # Invalid cross-nesting: discard the conflicting delimiter.
                pass
            else:
                stack.append(marker)
                result.append(marker)
            index += len(marker)
        result.extend(reversed(stack))
        return "".join(result)


@dataclass(frozen=True, slots=True)
class OopzTextBudget:
    """Centralized platform and project text limits."""

    safe_limit: int = 1950
    hard_limit: int = 2000

    def __post_init__(self) -> None:
        if self.safe_limit <= 0 or self.hard_limit < self.safe_limit:
            raise ValueError("OOPZ text limits are inconsistent")

    def assert_safe(self, text: str) -> None:
        if oopz_units(text) > self.safe_limit:
            raise ValueError("Rendered OOPZ message exceeds the safe text budget")


class OopzMessageRenderer:
    """Render complete snapshots; never expose raw tool payloads or exceptions."""

    max_visible_steps = 5

    def __init__(
        self,
        normalizer: OopzMarkupNormalizer | None = None,
        budget: OopzTextBudget | None = None,
    ) -> None:
        self._normalizer = normalizer or OopzMarkupNormalizer()
        self._budget = budget or OopzTextBudget()

    def render(self, state: AgentLoopViewState) -> str:
        if state.phase is DisplayPhase.SUCCEEDED:
            rendered = self._render_success(state)
        elif state.phase is DisplayPhase.FAILED:
            rendered = self._render_terminal(
                "⚠️ **失败了(┬┬﹏┬┬)**",
                state.terminal_message or "模型服务暂时不可用，请稍后再试。",
            )
        elif state.phase is DisplayPhase.CANCELLED:
            rendered = "⏹ **已取消当前回答**"
        else:
            rendered = self._render_active(state)
        if oopz_units(rendered) > self._budget.safe_limit:
            rendered = self._safe_plain_prefix(rendered)
        self._budget.assert_safe(rendered)
        return rendered

    def _render_active(self, state: AgentLoopViewState) -> str:
        header = {
            DisplayPhase.CREATED: "✨ **初音未来 正在准备回答…**",
            DisplayPhase.ACCEPTED: "✨ **初音未来 正在准备回答…**",
            DisplayPhase.THINKING: "♪ **初音未来 正在思考…**",
            DisplayPhase.TOOL_RUNNING: "🛠 **初音未来 正在处理…**",
            DisplayPhase.DRAFTING: "🎤 **初音未来 正在组织回答…**",
        }.get(state.phase, "♪ **初音未来 正在思考…**")
        lines = [header]
        lines.extend(self._step_lines(state))
        if state.current_draft:
            draft = self._normalizer.normalize(state.current_draft)
            prefix = "\n".join(lines)
            if draft:
                prefix += "\n\n"
                available = self._budget.safe_limit - oopz_units(prefix)
                lines = [prefix + self._truncate_tail(draft, available)]
        return "\n".join(lines)

    def _step_lines(self, state: AgentLoopViewState) -> list[str]:
        running = [step for step in state.steps if step.status is ToolStepStatus.RUNNING]
        failed = [step for step in state.steps if step.status is ToolStepStatus.FAILED]
        succeeded = [step for step in state.steps if step.status is ToolStepStatus.SUCCEEDED]
        if state.phase is DisplayPhase.DRAFTING:
            selected = running + failed[-1:]
            visible_limit = self.max_visible_steps - (1 if succeeded else 0)
            lines = self._render_steps(selected[:visible_limit])
            if succeeded:
                lines.append(f"✅ 已完成 {len(succeeded)} 个步骤")
            return lines

        selected = running[: self.max_visible_steps]
        remaining = self.max_visible_steps - len(selected)
        if remaining and failed:
            selected.extend(failed[-1:])
            remaining -= 1
        if remaining:
            selected.extend(reversed(succeeded[-remaining:]))
        hidden = len(state.steps) - len(selected)
        lines = self._render_steps(selected)
        if hidden > 0:
            lines.append(f"… 已折叠 {hidden} 个已完成步骤")
        return lines

    def _render_steps(self, steps: list[ToolStepView]) -> list[str]:
        return [line for step in steps for line in self._step_lines_for(step)]

    def _step_lines_for(self, step: ToolStepView) -> list[str]:
        name = self._normalizer.plain_text(step.display_name)[:48]
        if step.status is ToolStepStatus.RUNNING:
            header = f"⏳ *{name}…*"
        elif step.status is ToolStepStatus.FAILED:
            header = f"⚠️ {name}未完成，正在调整"
        else:
            header = f"✅ {name}"
        details = [detail for detail in (step.request_detail, step.result_detail) if detail]
        if not details:
            return [header]
        normalized = "；".join(self._normalizer.plain_text(detail) for detail in details)[:180]
        return [header, f"  ↳ {normalized}"]

    def _render_success(self, state: AgentLoopViewState) -> str:
        header = self._success_header(state)
        normalized = self._normalizer.normalize(state.final_text)
        if oopz_units(header + normalized) <= self._budget.safe_limit:
            return header + normalized
        plain = self._normalizer.plain_text(state.final_text)
        available = self._budget.safe_limit - oopz_units(header)
        return header + self._middle_fold(plain, available)

    @staticmethod
    def _success_header(state: AgentLoopViewState) -> str:
        statistics: list[str] = []
        if state.elapsed_seconds is not None:
            statistics.append(f"{state.elapsed_seconds:.1f}s")
        if state.tool_calls is not None:
            statistics.append(f"{state.tool_calls} 次工具")
        if state.input_tokens is not None or state.output_tokens is not None:
            total_tokens = (state.input_tokens or 0) + (state.output_tokens or 0)
            statistics.append(f"{OopzMessageRenderer._compact_number(total_tokens)} tokens")
        suffix = f" · {' · '.join(statistics)}" if statistics else ""
        return f"🎵 **初音未来**{suffix}\n"

    @staticmethod
    def _compact_number(value: int) -> str:
        if value < 1000:
            return str(value)
        rendered = f"{value / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{rendered}k"

    def _render_terminal(self, header: str, message: str) -> str:
        prefix = f"{header}\n"
        normalized = self._normalizer.normalize(message)
        return prefix + self._truncate_tail(
            normalized,
            self._budget.safe_limit - oopz_units(prefix),
        )

    def _middle_fold(self, text: str, available: int) -> str:
        marker = "\n…（中间内容因 OOPZ 长度限制已折叠）…\n"
        for _ in range(5):
            content_units = max(available - oopz_units(marker), 0)
            head = _take_prefix(text, int(content_units * 0.7))
            tail = _take_suffix(text[len(head) :], content_units - oopz_units(head))
            omitted = max(len(text) - len(head) - len(tail), 0)
            updated = f"\n…（中间 {omitted} 字因 OOPZ 长度限制已折叠）…\n"
            if updated == marker:
                break
            marker = updated
        content_units = max(available - oopz_units(marker), 0)
        head = _take_prefix(text, int(content_units * 0.7))
        tail = _take_suffix(text[len(head) :], content_units - oopz_units(head))
        return head + marker + tail

    def _truncate_tail(self, text: str, available: int) -> str:
        if oopz_units(text) <= available:
            return text
        plain = self._normalizer.plain_text(text)
        marker = "…"
        return _take_prefix(plain, max(available - oopz_units(marker), 0)) + marker

    def _safe_plain_prefix(self, text: str) -> str:
        plain = self._normalizer.plain_text(text)
        return self._truncate_tail(plain, self._budget.safe_limit)
