"""Bounded OOPZ rendering for tracked Agent diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from cywl_oopz.features.admin.models import AgentDiagnosticTool, AgentResponseDiagnostic
from cywl_oopz.features.agent.tool_progress import ToolProgressCatalog

from .message_renderer import OopzMarkupNormalizer, OopzTextBudget, oopz_units


class DiagnosticRedactor:
    """Recursively remove common credentials before verbose rendering."""

    _SENSITIVE = frozenset(
        {
            "api_key",
            "apikey",
            "authorization",
            "cookie",
            "credentials",
            "password",
            "private_key",
            "secret",
            "signed_url",
            "token",
        }
    )

    def redact(self, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): ("[已隐藏]" if self._sensitive(str(key)) else self.redact(item))
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return [self.redact(item) for item in value[:50]]
        if isinstance(value, str):
            return self._redact_url(value)
        return value

    def _sensitive(self, key: str) -> bool:
        normalized = key.casefold().replace("-", "_")
        return (
            normalized in self._SENSITIVE
            or any(
                part in normalized for part in ("authorization", "credential", "password", "secret")
            )
            or normalized.endswith(("_api_key", "_cookie", "_token"))
        )

    def _redact_url(self, value: str) -> str:
        if not value.casefold().startswith(("http://", "https://")):
            return value
        parsed = urlsplit(value)
        sensitive_query = any(
            self._sensitive(key) or key.casefold() in {"sign", "signature"}
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        )
        if not sensitive_query:
            return value
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "[已隐藏]", ""))


@dataclass(frozen=True, slots=True)
class OopzPaginator:
    """Split normalized text on useful boundaries under OOPZ's UTF-16 budget."""

    budget: OopzTextBudget = OopzTextBudget()
    max_pages: int = 8
    max_total_units: int = 12_000

    def __post_init__(self) -> None:
        if self.max_pages <= 0 or self.max_total_units <= 0:
            raise ValueError("Paginator limits must be positive")

    def paginate(self, title: str, body: str) -> tuple[str, ...]:
        title = title.strip()
        body = body.strip()
        if not title or not body:
            raise ValueError("Paginator title and body must not be empty")
        chunks: list[str] = []
        remaining = body
        total_units = 0
        while remaining and len(chunks) < self.max_pages:
            page_number = len(chunks) + 1
            header = f"{title} · {page_number}/{self.max_pages}\n"
            available = min(
                self.budget.safe_limit - oopz_units(header),
                self.max_total_units - total_units,
            )
            if available <= 0:
                break
            chunk, remaining = self._take_chunk(remaining, available)
            if not chunk:
                break
            chunks.append(chunk)
            total_units += oopz_units(chunk)
        if remaining:
            marker = "\n… 其余诊断内容已截断"
            last = chunks[-1] if chunks else ""
            allowed = self.budget.safe_limit - oopz_units(
                f"{title} · {max(len(chunks), 1)}/{max(len(chunks), 1)}\n"
            )
            while last and oopz_units(last + marker) > allowed:
                last = last[:-1]
            if chunks:
                chunks[-1] = last.rstrip() + marker
            else:
                chunks.append(marker.strip())
        total = len(chunks)
        pages = tuple(
            f"{title} · {index}/{total}\n{chunk}" for index, chunk in enumerate(chunks, start=1)
        )
        for page in pages:
            self.budget.assert_safe(page)
        return pages

    @staticmethod
    def _take_chunk(text: str, available: int) -> tuple[str, str]:
        used = 0
        end = 0
        last_line = 0
        last_paragraph = 0
        for index, character in enumerate(text):
            units = oopz_units(character)
            if used + units > available:
                break
            used += units
            end = index + 1
            if character == "\n":
                last_line = end
                if index > 0 and text[index - 1] == "\n":
                    last_paragraph = end
        if end == len(text):
            return text, ""
        split = last_paragraph or last_line or end
        return text[:split].rstrip(), text[split:].lstrip()


class OopzAgentDiagnosticRenderer:
    """Render safe normal/verbose diagnostics independently from database I/O."""

    max_payload_characters = 2_000

    def __init__(
        self,
        display_names: Mapping[str, str] | None = None,
        *,
        normalizer: OopzMarkupNormalizer | None = None,
        redactor: DiagnosticRedactor | None = None,
        paginator: OopzPaginator | None = None,
    ) -> None:
        self._display_names = dict(display_names or {})
        self._normalizer = normalizer or OopzMarkupNormalizer()
        self._redactor = redactor or DiagnosticRedactor()
        self._paginator = paginator or OopzPaginator()
        self._progress = ToolProgressCatalog()

    def render(
        self,
        diagnostic: AgentResponseDiagnostic,
        *,
        verbose: bool,
    ) -> tuple[str, ...]:
        snapshot = diagnostic.receipt.diagnostic_snapshot
        lines = [self._summary(diagnostic, snapshot)]
        if diagnostic.provider_alias or diagnostic.model_alias:
            lines.append(f"模型：{diagnostic.provider_alias}/{diagnostic.model_alias}")
        retries = self._integer(snapshot.get("provider_retry_count"))
        if retries:
            lines.append(f"↻ 上游重试 {retries} 次")
        steps = snapshot.get("steps")
        if isinstance(steps, list) and steps:
            lines.extend(self._snapshot_steps(steps))
        elif diagnostic.tools:
            lines.extend(self._persisted_steps(diagnostic.tools))
        else:
            lines.append("\n没有工具调用。")

        answer = diagnostic.assistant_text or self._string(snapshot.get("final_text"))
        if answer:
            lines.append("\n**完整回答**\n" + self._normalizer.normalize(answer))
        elif diagnostic.status == "running":
            lines.append("\n回复仍在运行，完整回答尚未生成。")
        elif diagnostic.run_id is None:
            lines.append("\n回复刚刚开始，尚无运行记录。")
        else:
            failure = self._string(snapshot.get("terminal_message")) or diagnostic.error_code
            lines.append(f"\n没有生成完整回答。{(' ' + failure) if failure else ''}")

        if verbose:
            lines.extend(self._verbose(diagnostic, snapshot))
        body = "\n".join(lines)
        return self._paginator.paginate("🔎 **Agent 回复详情**", body)

    def _summary(self, diagnostic: AgentResponseDiagnostic, snapshot: Mapping[str, Any]) -> str:
        status = diagnostic.status or self._string(snapshot.get("phase")) or "unknown"
        label = {
            "succeeded": "成功",
            "running": "仍在运行",
            "failed": "失败",
            "cancelled": "已取消",
            "abandoned": "已中断",
        }.get(status, status)
        facts = [label]
        elapsed = self._number(snapshot.get("elapsed_seconds"))
        if elapsed is None and diagnostic.started_at is not None:
            end = diagnostic.finished_at or datetime.now(diagnostic.started_at.tzinfo)
            elapsed = max((end - diagnostic.started_at).total_seconds(), 0)
        if elapsed is not None:
            facts.append(f"{elapsed:.1f}s")
        requests = self._metric(diagnostic, snapshot, "model_requests")
        tools = self._metric(diagnostic, snapshot, "tool_calls")
        input_tokens = self._metric(diagnostic, snapshot, "input_tokens")
        output_tokens = self._metric(diagnostic, snapshot, "output_tokens")
        if requests is not None:
            facts.append(f"{requests} 次请求")
        if tools is not None:
            facts.append(f"{tools} 次工具")
        if input_tokens is not None or output_tokens is not None:
            facts.append(f"{input_tokens or 0}→{output_tokens or 0} tokens")
        return " · ".join(facts)

    def _snapshot_steps(self, steps: list[object]) -> list[str]:
        lines = ["\n**工具步骤**"]
        for index, raw in enumerate(steps, start=1):
            if not isinstance(raw, Mapping):
                continue
            status = self._string(raw.get("status"))
            icon = {"succeeded": "✅", "failed": "⚠️", "running": "⏳"}.get(status, "•")
            name = self._line(raw.get("display_name"), 48) or self._line(raw.get("tool_name"), 48)
            header = f"{index}. {icon} **{name or '执行操作'}**"
            subject = self._line(raw.get("subject"), 80)
            summary = self._line(raw.get("summary"), 100)
            if subject:
                header += f" {subject}"
            if summary:
                header += f" · {summary}"
            lines.append(header)
            details = raw.get("items") or raw.get("preview_lines")
            if isinstance(details, list):
                lines.extend(f"   • {self._line(item, 180)}" for item in details[:3] if item)
        return lines

    def _persisted_steps(self, tools: tuple[AgentDiagnosticTool, ...]) -> list[str]:
        lines = ["\n**工具步骤**"]
        for index, tool in enumerate(tools, start=1):
            succeeded = tool.status == "succeeded"
            if tool.status == "started":
                icon = "⏳"
            elif succeeded:
                icon = "✅"
            else:
                icon = "⚠️"
            request = self._progress.request(tool.name, tool.input_payload)
            result = self._progress.result(
                tool.name,
                {"data": tool.output_payload} if succeeded else {"error": tool.error_code},
                succeeded=succeeded,
            )
            name = self._display_names.get(tool.name, tool.name)
            header = f"{index}. {icon} **{self._line(name, 48)}**"
            subject = result.subject or request.subject
            if subject:
                header += f" {subject}"
            if result.summary:
                header += f" · {result.summary}"
            duration = self._duration(tool)
            if duration is not None:
                header += f" · {duration:.1f}s"
            lines.append(header)
            details = result.items or result.preview_lines
            lines.extend(f"   • {item}" for item in details)
        return lines

    def _verbose(
        self,
        diagnostic: AgentResponseDiagnostic,
        snapshot: Mapping[str, Any],
    ) -> list[str]:
        lines = ["\n**Verbose 诊断**"]
        if diagnostic.run_id is not None:
            lines.append(f"run：{str(diagnostic.run_id)[:8]}")
        if diagnostic.thread_id is not None:
            lines.append(f"thread：{str(diagnostic.thread_id)[:8]}")
        if diagnostic.selection_source:
            lines.append(f"选择来源：{diagnostic.selection_source}")
        if diagnostic.stop_reason:
            lines.append(f"停止原因：{diagnostic.stop_reason}")
        if diagnostic.error_code:
            lines.append(f"错误码：{diagnostic.error_code}")
        if diagnostic.limits:
            lines.append("limits：" + self._json(diagnostic.limits))
        if diagnostic.usage:
            lines.append("usage：" + self._json(diagnostic.usage))
        retries = snapshot.get("provider_retries")
        if isinstance(retries, list):
            for retry in retries[:8]:
                if isinstance(retry, Mapping):
                    lines.append("retry：" + self._json(retry))
        for index, tool in enumerate(diagnostic.tools, start=1):
            duration = self._duration(tool)
            duration_text = f" · {duration:.3f}s" if duration is not None else ""
            lines.append(
                f"\n{index}. {tool.name}@{tool.version} · {tool.effect} · "
                f"{tool.status}{duration_text}"
            )
            if tool.error_code:
                lines.append(f"error：{tool.error_code}")
            lines.append("input：" + self._json(tool.input_payload))
            if tool.output_payload is not None:
                lines.append("output：" + self._json(tool.output_payload))
        return lines

    def _json(self, value: object) -> str:
        text = json.dumps(
            self._redactor.redact(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if len(text) > self.max_payload_characters:
            return text[: self.max_payload_characters - 1] + "…"
        return text

    @staticmethod
    def _duration(tool: AgentDiagnosticTool) -> float | None:
        if tool.finished_at is None:
            return None
        return max((tool.finished_at - tool.started_at).total_seconds(), 0)

    @classmethod
    def _metric(
        cls,
        diagnostic: AgentResponseDiagnostic,
        snapshot: Mapping[str, Any],
        name: str,
    ) -> int | None:
        value = snapshot.get(name)
        if value is None:
            value = diagnostic.usage.get(name)
        return cls._integer(value)

    @staticmethod
    def _integer(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return max(int(value), 0)

    @staticmethod
    def _number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return max(float(value), 0)

    @staticmethod
    def _string(value: object) -> str:
        return value if isinstance(value, str) else ""

    def _line(self, value: object, limit: int) -> str:
        text = self._normalizer.plain_text(str(value)) if value is not None else ""
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"
