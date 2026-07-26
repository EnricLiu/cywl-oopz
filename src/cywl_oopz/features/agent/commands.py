"""OOPZ command controller for database-backed Provider switching."""

from __future__ import annotations

from uuid import UUID

from oopz_sdk.events.context import EventContext

from cywl_oopz.commands.router import ParsedCommand
from cywl_oopz.features.chat.commands import ChatCommandController
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor

from .memory import (
    MemoryCapacityError,
    MemoryDisabledError,
    MemoryItemTooLongError,
    MemoryService,
)
from .service import AgentConversationService


class ProviderCommand(ChatCommandController):
    """Show or switch the current Agent Provider/model."""

    name = "provider"
    description = "查看、列出或切换 Agent Provider/模型。"

    def __init__(
        self,
        service: AgentConversationService,
        tasks: ChatTaskSupervisor,
    ) -> None:
        super().__init__(service)
        self._agent = service
        self._tasks = tasks

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        try:
            key = self._key(context)
            if not command.arguments:
                selection = await self._agent.current_selection(key)
                await context.reply(
                    "当前 Provider/模型："
                    f"{selection.model.provider_alias}/{selection.model.model_alias}"
                    f"（来源：{selection.source.value}）"
                )
                return

            action, *arguments = command.arguments
            if action == "list" and not arguments:
                models = self._agent.list_models()
                message = "可用 Provider/模型：\n" + (
                    "\n".join(f"- {model}" for model in models)
                    if models
                    else "（当前没有可选模型）"
                )
                await context.reply(message)
                return

            if action not in {"use", "default"} or not 1 <= len(arguments) <= 2:
                await context.reply(
                    "用法：!provider [list|use <provider> [model]|default <provider> [model]]"
                )
                return
            if self._tasks.has_active(key):
                await context.reply("当前正在生成回复；请等待完成或先使用 !cancel。")
                return

            selected = await self._agent.select_provider(
                key,
                arguments[0],
                arguments[1] if len(arguments) == 2 else None,
                user_default=action == "default",
            )
        except Exception as exc:
            await self._reply_error(context, exc)
            return

        if action == "default":
            await context.reply(f"后续新对话的默认模型已切换为：{selected}")
        else:
            await context.reply(f"当前对话模型已切换为：{selected}")


class ToolsCommand(ChatCommandController):
    """List the exact server-authorized tools visible in this conversation."""

    name = "tools"
    description = "查看当前 Agent 实际可用的工具。"

    def __init__(self, service: AgentConversationService) -> None:
        super().__init__(service)
        self._agent = service

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        if command.arguments:
            await context.reply("用法：!tools")
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


class MemoryCommand(ChatCommandController):
    """Let users inspect and control only their own long-term memory."""

    name = "memory"
    description = "查看、保存、关闭或删除自己的长期记忆。"

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
                    else "长期记忆已关闭；已有内容仍保留，可使用 !memory forget all 删除。"
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
                "用法：!memory [status|list|on|off|remember <内容>|forget <ID|all>]"
            )
        except MemoryDisabledError:
            await context.reply("长期记忆当前已关闭；请先使用 !memory on。")
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
