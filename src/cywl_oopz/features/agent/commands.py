"""OOPZ command controller for database-backed Provider switching."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from uuid import UUID

from oopz_sdk.events.context import EventContext

from cywl_oopz.commands.router import ParsedCommand
from cywl_oopz.core.lifecycle import ModelSelectionSource
from cywl_oopz.core.observability import exception_kind
from cywl_oopz.features.chat.commands import ChatCommandController
from cywl_oopz.features.chat.models import (
    ChatInvocation,
    ChatInvocationFactory,
    ConversationKey,
)
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor

from .direct_tools import DirectToolService
from .memory import (
    MemoryCapacityError,
    MemoryDisabledError,
    MemoryItemTooLongError,
    MemoryService,
)
from .models import ModelSelection, SelectableModel
from .service import AgentConversationService
from .skills.library import AgentSkillLibraryService

logger = logging.getLogger(__name__)


class ModelCommandView:
    """Render compact OOPZ-safe Provider/model menus with stable command aliases."""

    max_characters = 1900
    _source_labels = {
        ModelSelectionSource.THREAD: "当前对话",
        ModelSelectionSource.USER: "个人默认",
        ModelSelectionSource.CHANNEL: "频道默认",
        ModelSelectionSource.APPLICATION: "系统默认",
    }

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def providers(
        self,
        selection: ModelSelection,
        choices: tuple[SelectableModel, ...],
    ) -> str:
        """Show the current selection and group model aliases by Provider."""
        groups: dict[tuple[str, str], list[SelectableModel]] = defaultdict(list)
        for choice in choices:
            groups[(choice.provider_alias, choice.provider_display_name)].append(choice)
        footer = (
            "",
            f"切换当前对话：{self._prefix}provider <Provider> [模型]",
            f"设为个人默认：{self._prefix}provider default <Provider> [模型]",
        )
        lines = [self._current(selection), "", "**可用 Provider**"]
        for (alias, display_name), models in groups.items():
            model_labels = "、".join(
                f"{model.model_alias}{'（默认）' if model.is_provider_default else ''}"
                for model in models
            )
            line = f"- **{alias}** {self._compact_name(display_name)}：{model_labels}"
            if not self._can_append(lines, line, footer):
                lines.append("…其余 Provider 已省略")
                break
            lines.append(line)
        if not groups:
            lines.append("（当前没有可选模型）")
        return "\n".join((*lines, *footer))

    def models(
        self,
        selection: ModelSelection,
        choices: tuple[SelectableModel, ...],
    ) -> str:
        """Show models belonging to the currently selected Provider."""
        current_provider = selection.model.provider_alias
        provider_choices = tuple(
            choice for choice in choices if choice.provider_alias == current_provider
        )
        footer = (
            "",
            f"切换模型：{self._prefix}model <模型>",
            f"跨 Provider：{self._prefix}model <Provider>/<模型>",
            f"查看全部 Provider：{self._prefix}provider",
        )
        lines = [self._current(selection), "", f"**{current_provider} 可用模型**"]
        for choice in provider_choices:
            labels = []
            if choice.model_display_name != choice.model_alias:
                labels.append(self._compact_name(choice.model_display_name))
            if choice.is_provider_default:
                labels.append("Provider 默认")
            suffix = f" · {' · '.join(labels)}" if labels else ""
            line = f"- **{choice.model_alias}**{suffix}"
            if not self._can_append(lines, line, footer):
                lines.append("…其余模型已省略")
                break
            lines.append(line)
        if not provider_choices:
            lines.append("（当前 Provider 没有其他可选模型）")
        return "\n".join((*lines, *footer))

    def provider_usage(self) -> str:
        """Return a short syntax guide including the preferred shorthand."""
        return "\n".join(
            (
                "**Provider 命令**",
                f"- {self._prefix}provider：查看当前选择和全部 Provider",
                f"- {self._prefix}provider <Provider> [模型]：切换当前对话",
                f"- {self._prefix}provider default <Provider> [模型]：设置个人默认",
                f"- 旧写法 {self._prefix}provider use ... 仍然可用",
            )
        )

    def model_usage(self) -> str:
        """Return the simple current-Provider model syntax."""
        return "\n".join(
            (
                "**模型命令**",
                f"- {self._prefix}model：查看当前 Provider 的模型",
                f"- {self._prefix}model <模型>：在当前 Provider 内切换",
                f"- {self._prefix}model <Provider>/<模型>：跨 Provider 切换",
            )
        )

    def active_run_message(self) -> str:
        """Explain how to cancel an in-flight run with the configured prefix."""
        return f"当前正在生成回复；请等待完成或先使用 {self._prefix}cancel。"

    @classmethod
    def _current(cls, selection: ModelSelection) -> str:
        source = cls._source_labels.get(selection.source, selection.source.value)
        return (
            f"🎛️ **当前模型** {selection.model.provider_alias}/"
            f"{selection.model.model_alias} · {source}"
        )

    @staticmethod
    def _compact_name(value: str) -> str:
        return value if len(value) <= 80 else f"{value[:77]}..."

    def _can_append(
        self,
        lines: list[str],
        line: str,
        footer: tuple[str, ...],
    ) -> bool:
        return len("\n".join((*lines, line, *footer))) <= self.max_characters


class AgentModelCommand(ChatCommandController):
    """Inspect or switch models with the current Provider as the default scope."""

    name = "model"
    description = "查看当前 Provider 的模型，或切换当前对话模型。"
    category = "对话"
    usage = ("model [list|help|模型编号或名称]",)

    def __init__(
        self,
        service: AgentConversationService,
        tasks: ChatTaskSupervisor,
        prefix: str = "/",
    ) -> None:
        super().__init__(service)
        self._agent = service
        self._tasks = tasks
        self._view = ModelCommandView(prefix)

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        key = self._key(context)
        action = command.arguments[0].casefold() if command.arguments else ""
        if not command.arguments or (action == "list" and len(command.arguments) == 1):
            await self._show_models(context, key)
            return
        if action == "help" and len(command.arguments) == 1:
            await context.reply(self._view.model_usage())
            return
        if len(command.arguments) != 1 or action in {"help", "list"}:
            await context.reply(self._view.model_usage())
            return
        if self._tasks.has_active(key):
            await context.reply(self._view.active_run_message())
            return
        choice = command.arguments[0]
        try:
            selected = await self._agent.select_model(key, choice)
        except ValueError:
            await context.reply(f"没有找到可选模型「{choice}」。\n{self._view.model_usage()}")
            return
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        await context.reply(f"✅ **当前对话模型** {selected}")

    async def _show_models(self, context: EventContext, key: ConversationKey) -> None:
        try:
            catalog = await self._agent.model_catalog_view(key)
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        await context.reply(self._view.models(catalog.selection, catalog.choices))


class ProviderCommand(ChatCommandController):
    """Show or switch the current Agent Provider/model."""

    name = "provider"
    description = "查看、列出或切换 Agent Provider/模型。"
    category = "Agent"
    usage = (
        "provider [list|help]",
        "provider [use] <Provider> [模型]",
        "provider default <Provider> [模型]",
    )

    def __init__(
        self,
        service: AgentConversationService,
        tasks: ChatTaskSupervisor,
        prefix: str = "/",
    ) -> None:
        super().__init__(service)
        self._agent = service
        self._tasks = tasks
        self._view = ModelCommandView(prefix)

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        key = self._key(context)
        action = command.arguments[0].casefold() if command.arguments else ""
        if not command.arguments or (action == "list" and len(command.arguments) == 1):
            await self._show_providers(context, key)
            return
        if action == "help" and len(command.arguments) == 1:
            await context.reply(self._view.provider_usage())
            return
        if action in {"help", "list"}:
            await context.reply(self._view.provider_usage())
            return

        arguments = command.arguments[1:] if action in {"use", "default"} else command.arguments
        user_default = action == "default"
        parsed = self._selection_arguments(arguments)
        if parsed is None:
            await context.reply(self._view.provider_usage())
            return
        provider_alias, model_alias = parsed
        if self._tasks.has_active(key):
            await context.reply(self._view.active_run_message())
            return
        try:
            selected = await self._agent.select_provider(
                key,
                provider_alias,
                model_alias,
                user_default=user_default,
            )
        except ValueError:
            target = provider_alias + (f"/{model_alias}" if model_alias else "")
            await context.reply(
                f"没有找到可选的 Provider/模型「{target}」。\n{self._view.provider_usage()}"
            )
            return
        except Exception as exc:
            await self._reply_error(context, exc)
            return

        if user_default:
            await context.reply(
                f"✅ **个人默认模型** {selected}\n"
                "未单独选择模型的对话会使用它；当前对话的独立选择不会被覆盖。"
            )
        else:
            await context.reply(f"✅ **当前对话模型** {selected}")

    async def _show_providers(self, context: EventContext, key: ConversationKey) -> None:
        try:
            catalog = await self._agent.model_catalog_view(key)
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        await context.reply(self._view.providers(catalog.selection, catalog.choices))

    @staticmethod
    def _selection_arguments(arguments: tuple[str, ...]) -> tuple[str, str | None] | None:
        if not 1 <= len(arguments) <= 2:
            return None
        if len(arguments) == 2:
            return arguments[0], arguments[1]
        value = arguments[0]
        if "/" not in value:
            return value, None
        provider_alias, model_alias = value.split("/", 1)
        if not provider_alias or not model_alias:
            return None
        return provider_alias, model_alias


class ToolsCommand(ChatCommandController):
    """List the exact server-authorized tools visible in this conversation."""

    name = "tools"
    description = "查看当前 Agent 实际可用的工具。"
    category = "Agent"
    usage = ("tools",)

    def __init__(self, service: AgentConversationService) -> None:
        super().__init__(service)
        self._agent = service

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        if command.arguments:
            await context.reply("用法：/tools")
            return
        try:
            tools = await self._agent.available_tools(self._key(context))
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        if not tools:
            await context.reply("当前对话没有可用的 Agent 工具。")
            return
        effects = {"read": "只读", "write": "写操作", "admin": "管理"}
        lines = ["当前可用 Agent 工具："]
        lines.extend(
            f"- {tool.name}（{effects[tool.effect.value]}）：{tool.description}" for tool in tools
        )
        await context.reply("\n".join(lines))


class SkillsCommand(ChatCommandController):
    """List available Skills and caller-scoped sharing state."""

    name = "skills"
    description = "查看当前 Agent 可按需加载的技能。"
    category = "Agent"
    usage = ("skills [owned|shared|invitations]",)
    max_reply_characters = 1900

    def __init__(
        self,
        service: AgentConversationService,
        library: AgentSkillLibraryService | None = None,
    ) -> None:
        super().__init__(service)
        self._agent = service
        self._library = library

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        action = command.arguments[0].casefold() if len(command.arguments) == 1 else ""
        if command.arguments and action not in {"owned", "shared", "invitations"}:
            await context.reply("用法：/skills [owned|shared|invitations]")
            return
        if action:
            await self._show_library_group(action, context)
            return
        try:
            skills = await self._agent.available_skills(self._key(context))
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        if not skills:
            await context.reply("当前对话没有可用的 Agent 技能。")
            return

        access_labels = {
            "builtin": "内置",
            "owned": "我的",
            "shared": "共享",
        }
        lines = ["当前可用 Agent 技能（需要时由模型按需加载）："]
        for index, skill in enumerate(skills):
            description = (
                skill.description
                if len(skill.description) <= 240
                else f"{skill.description[:237]}..."
            )
            access = access_labels[skill.access.value]
            line = (
                f"- **{skill.display_name}** {skill.name} · {access} · "
                f"v{skill.version}：{description}"
            )
            omitted = len(skills) - index
            suffix = f"\n…另有 {omitted} 个技能未显示。" if omitted else ""
            if len("\n".join((*lines, line))) + len(suffix) > self.max_reply_characters:
                lines.append(f"…另有 {omitted} 个技能未显示。")
                break
            lines.append(line)
        await context.reply("\n".join(lines))

    async def _show_library_group(
        self,
        action: str,
        context: EventContext,
    ) -> None:
        if self._library is None:
            await context.reply("当前未启用个人技能库。")
            return
        try:
            person_id = self._key(context).person_id
            library = await self._library.library(person_id)
        except Exception as exc:
            await self._reply_error(context, exc)
            return
        if action == "owned":
            lines = ["**我的技能**"]
            if not library.owned:
                lines.append("（暂无）")
            else:
                lines.extend(
                    (
                        f"- **{item.discovery.display_name}** "
                        f"{item.discovery.name} · "
                        f"{'使用中' if item.active else '已归档'}"
                    )
                    for item in library.owned
                )
        elif action == "shared":
            lines = ["**已接受的共享技能**"]
            if not library.shared:
                lines.append("（暂无）")
            else:
                lines.extend(
                    f"- **{skill.display_name}** {skill.name} · v{skill.version}"
                    for skill in library.shared
                )
        else:
            lines = ["**待处理的技能邀请**"]
            if not library.pending_invitations:
                lines.append("（暂无）")
            else:
                lines.extend(
                    (
                        f"- **{item.skill.display_name}** · 邀请 ID "
                        f"{item.share.id}"
                        f"{'' if item.active else ' · 当前已归档'}"
                    )
                    for item in library.pending_invitations
                )
                lines.append("可直接告诉未来接受或拒绝其中一项。")
        await context.reply(self._bounded(lines))

    def _bounded(self, lines: list[str]) -> str:
        rendered: list[str] = []
        for index, line in enumerate(lines):
            remaining = len(lines) - index
            omitted = f"\n…另有 {remaining} 项未显示。" if remaining else ""
            if len("\n".join((*rendered, line))) + len(omitted) > self.max_reply_characters:
                rendered.append(f"…另有 {remaining} 项未显示。")
                break
            rendered.append(line)
        return "\n".join(rendered)


class ToolCommand:
    """Describe or directly execute one Agent tool for development debugging."""

    name = "tool"
    description = "直接执行 Agent 工具，或查看指定工具的 JSON Schema。"
    category = "Agent"
    usage = ("tool <tool-id> <JSON对象|--help>",)
    max_reply_characters = 1900

    def __init__(
        self,
        service: DirectToolService,
        prefix: str,
        invocation_factory: ChatInvocationFactory | None = None,
    ) -> None:
        self._service = service
        self._prefix = prefix
        self._invocations = invocation_factory

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        parts = self._raw_arguments(command).split(maxsplit=1)
        if not parts:
            await self._reply(
                context,
                self._usage_error(),
            )
            return

        tool_name = parts[0].casefold()
        body = parts[1] if len(parts) == 2 else ""
        description = self._service.describe(tool_name)
        if description is None:
            await self._reply(
                context,
                {"ok": False, "error": "tool_not_registered"},
            )
            return
        if body == "--help":
            await self._reply(context, {"ok": True, "data": description})
            return
        if not body:
            await self._reply(context, self._usage_error())
            return

        try:
            arguments = json.loads(body)
        except json.JSONDecodeError as exc:
            await self._reply(
                context,
                {
                    "ok": False,
                    "error": "invalid_json",
                    "line": exc.lineno,
                    "column": exc.colno,
                },
            )
            return
        if not isinstance(arguments, dict):
            await self._reply(
                context,
                {"ok": False, "error": "json_body_must_be_object"},
            )
            return

        try:
            result = await self._service.execute(
                ConversationKey.from_oopz_context(context),
                (
                    self._invocations.from_context(context)
                    if self._invocations is not None
                    else ChatInvocation.from_oopz_context(context)
                ),
                tool_name,
                arguments,
            )
        except Exception as exc:
            logger.warning(
                "Direct Agent tool command failed: tool=%s error=%s",
                tool_name,
                exception_kind(exc),
            )
            await self._reply(
                context,
                {"ok": False, "error": "tool_debug_unavailable"},
            )
            return
        await self._reply(context, result.model_payload())

    def _usage_error(self) -> dict[str, object]:
        return {
            "ok": False,
            "error": "usage",
            "usage": f"{self._prefix}tool <tool-id> <json-body|--help>",
        }

    @staticmethod
    def _raw_arguments(command: ParsedCommand) -> str:
        return command.raw_arguments.strip() or " ".join(command.arguments)

    @classmethod
    async def _reply(cls, context: EventContext, payload: dict[str, object]) -> None:
        await context.reply(cls._render(payload))

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


class MemoryCommand(ChatCommandController):
    """Let users inspect and control only their own long-term memory."""

    name = "memory"
    description = "查看、保存、关闭或删除自己的长期记忆。"
    category = "Agent"
    usage = (
        "memory [status|list|on|off]",
        "memory remember <内容>",
        "memory forget <ID|all>",
    )

    def __init__(
        self,
        chat: AgentConversationService,
        memory: MemoryService,
    ) -> None:
        super().__init__(chat)
        self._memory = memory

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        try:
            person_id = self._key(context).person_id
            if not command.arguments or command.arguments == ("status",):
                status = await self._memory.status(person_id)
                await context.reply(
                    "长期记忆："
                    f"{'已开启' if status.enabled else '已关闭'}；"
                    f"当前 {status.item_count} 条。"
                )
                return

            action, *arguments = command.arguments
            action = action.casefold()
            if action == "list" and not arguments:
                await self._list(person_id, context)
                return
            if action in {"on", "off"} and not arguments:
                enabled = action == "on"
                await self._memory.set_enabled(person_id, enabled)
                message = (
                    "长期记忆已开启。"
                    if enabled
                    else "长期记忆已关闭；已有内容仍保留，可使用 /memory forget all 删除。"
                )
                await context.reply(message)
                return
            if action == "remember" and arguments:
                item = await self._memory.remember(person_id, " ".join(arguments))
                await context.reply(f"已记住。记忆 ID：{item.id}")
                return
            if action == "forget" and len(arguments) == 1:
                if arguments[0].casefold() == "all":
                    count = await self._memory.forget_all(person_id)
                    await context.reply(f"已删除 {count} 条长期记忆。")
                    return
                try:
                    item_id = UUID(arguments[0])
                except ValueError:
                    await context.reply("记忆 ID 格式不正确。")
                    return
                deleted = await self._memory.forget(person_id, item_id)
                await context.reply("已删除该记忆。" if deleted else "没有找到该记忆。")
                return
            await context.reply(
                "用法：/memory [status|list|on|off|remember <内容>|forget <ID|all>]"
            )
        except MemoryDisabledError:
            await context.reply("长期记忆当前已关闭；请先使用 /memory on。")
        except MemoryCapacityError:
            await context.reply("长期记忆条目已满；请先删除不需要的内容。")
        except MemoryItemTooLongError:
            await context.reply("这条记忆太长，请缩短后再保存。")
        except Exception as exc:
            await self._reply_error(context, exc)

    async def _list(self, person_id: str, context: EventContext) -> None:
        items = await self._memory.list(person_id)
        if not items:
            await context.reply("当前没有长期记忆。")
            return
        lines = ["你的长期记忆："]
        for item in items:
            text = item.content.get("text")
            if isinstance(text, str):
                display = text if len(text) <= 240 else f"{text[:237]}..."
                lines.append(f"- {item.id}: {display}")
        await context.reply("\n".join(lines))
