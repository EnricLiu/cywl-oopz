"""Typed Agent tool discovery and direct debugging commands."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    CommandDefinition,
    CommandExecutionPolicy,
    ExecutionMode,
    NoArguments,
    NoArgumentsParser,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.models import CommandRequest
from cywl_oopz.core.observability import exception_kind

from .command_support import AgentCommandContext, AgentCommandErrorPresenter
from .direct_tools import DirectToolService
from .service import AgentConversationService

logger = logging.getLogger(__name__)


class ToolsCommand:
    name = "tools"
    description = "查看当前 Agent 实际可用的工具。"
    category = "Agent"
    usage = ("tools",)

    def __init__(self, service: AgentConversationService) -> None:
        self._agent = service
        self._errors = AgentCommandErrorPresenter()

    def definition(self) -> CommandDefinition[NoArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            NoArgumentsParser(),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=30.0),
        )

    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del arguments
        try:
            tools = await self._agent.available_tools(AgentCommandContext.conversation(request))
        except Exception as exc:
            await self._errors.reply(request, exc)
            return
        if not tools:
            await request.responder.reply("当前对话没有可用的 Agent 工具。")
            return
        effects = {"read": "只读", "write": "写操作", "admin": "管理"}
        lines = ["当前可用 Agent 工具："]
        lines.extend(
            f"- {tool.name}（{effects[tool.effect.value]}）：{tool.description}" for tool in tools
        )
        await request.responder.reply("\n".join(lines))


class DirectToolAction(StrEnum):
    HELP = "help"
    EXECUTE = "execute"
    INVALID = "invalid"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DirectToolArguments:
    action: DirectToolAction
    tool_name: str = ""
    arguments: Mapping[str, Any] = MappingProxyType({})
    error: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(self, "error", MappingProxyType(dict(self.error)))


class DirectToolArgumentsParser:
    """Parse raw JSON tail without reconstructing it from tokens."""

    def __init__(self, service: DirectToolService, prefix: str) -> None:
        self._service = service
        self._prefix = prefix

    def parse(self, request: CommandRequest) -> DirectToolArguments:
        assert request.text is not None
        parts = request.text.raw_tail.strip().split(maxsplit=1)
        if not parts:
            return self._invalid(self._usage_error())
        tool_name = parts[0].casefold()
        body = parts[1] if len(parts) == 2 else ""
        if self._service.describe(tool_name) is None:
            return DirectToolArguments(
                DirectToolAction.UNKNOWN,
                tool_name,
                error={"ok": False, "error": "tool_not_registered"},
            )
        if body == "--help":
            return DirectToolArguments(DirectToolAction.HELP, tool_name)
        if not body:
            return self._invalid(self._usage_error(), tool_name)
        try:
            arguments = json.loads(body)
        except json.JSONDecodeError as exc:
            return self._invalid(
                {
                    "ok": False,
                    "error": "invalid_json",
                    "line": exc.lineno,
                    "column": exc.colno,
                },
                tool_name,
            )
        if not isinstance(arguments, dict):
            return self._invalid(
                {"ok": False, "error": "json_body_must_be_object"},
                tool_name,
            )
        return DirectToolArguments(DirectToolAction.EXECUTE, tool_name, arguments)

    def _usage_error(self) -> dict[str, object]:
        return {
            "ok": False,
            "error": "usage",
            "usage": f"{self._prefix}tool <tool-id> <json-body|--help>",
        }

    @staticmethod
    def _invalid(
        error: Mapping[str, object],
        tool_name: str = "",
    ) -> DirectToolArguments:
        return DirectToolArguments(DirectToolAction.INVALID, tool_name, error=error)


class ToolCommand:
    name = "tool"
    description = "直接执行 Agent 工具，或查看指定工具的 JSON Schema。"
    category = "Agent"
    usage = ("tool <tool-id> <JSON对象|--help>",)
    max_reply_characters = 1900

    def __init__(self, service: DirectToolService, prefix: str) -> None:
        self._service = service
        self._parser = DirectToolArgumentsParser(service, prefix)

    def definition(self) -> CommandDefinition[DirectToolArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            self._parser,
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=90.0),
        )

    async def handle(self, request: CommandRequest, arguments: DirectToolArguments) -> None:
        if arguments.action in {DirectToolAction.INVALID, DirectToolAction.UNKNOWN}:
            await request.responder.reply(self._render(dict(arguments.error)))
            return
        if arguments.action is DirectToolAction.HELP:
            description = self._service.describe(arguments.tool_name)
            assert description is not None
            await request.responder.reply(self._render({"ok": True, "data": description}))
            return
        try:
            result = await self._service.execute(
                AgentCommandContext.conversation(request),
                AgentCommandContext.invocation(request),
                arguments.tool_name,
                dict(arguments.arguments),
            )
        except Exception as exc:
            logger.warning(
                "Direct Agent tool command failed: tool=%s error=%s",
                arguments.tool_name,
                exception_kind(exc),
            )
            await request.responder.reply(
                self._render({"ok": False, "error": "tool_debug_unavailable"})
            )
            return
        await request.responder.reply(self._render(result.model_payload()))

    @classmethod
    def _render(cls, payload: dict[str, object]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= cls.max_reply_characters:
            return encoded
        preview = encoded[: cls.max_reply_characters // 2]
        truncated: dict[str, object] = {
            "ok": payload.get("ok") is True,
            "truncated": True,
            "preview": preview,
        }
        rendered = json.dumps(truncated, ensure_ascii=False, separators=(",", ":"))
        while len(rendered) > cls.max_reply_characters and preview:
            preview = preview[: len(preview) // 2]
            truncated["preview"] = preview
            rendered = json.dumps(truncated, ensure_ascii=False, separators=(",", ":"))
        return rendered
