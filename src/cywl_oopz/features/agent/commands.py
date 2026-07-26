"""OOPZ command controller for database-backed Provider switching."""

from __future__ import annotations

from oopz_sdk.events.context import EventContext

from cywl_oopz.commands.router import ParsedCommand
from cywl_oopz.features.chat.commands import ChatCommandController
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor

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
