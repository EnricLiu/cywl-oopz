"""Typed user-owned long-term memory command."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    CommandDefinition,
    CommandExecutionPolicy,
    CommandUsageError,
    ExecutionMode,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.models import CommandRequest

from .command_support import AgentCommandErrorPresenter
from .memory import (
    MemoryCapacityError,
    MemoryDisabledError,
    MemoryItemTooLongError,
    MemoryService,
)


class MemoryAction(StrEnum):
    STATUS = "status"
    LIST = "list"
    ENABLE = "enable"
    REMEMBER = "remember"
    FORGET = "forget"
    FORGET_ALL = "forget_all"


@dataclass(frozen=True, slots=True)
class MemoryArguments:
    action: MemoryAction
    enabled: bool | None = None
    text: str = ""
    item_id: UUID | None = None


class MemoryArgumentsParser:
    def parse(self, request: CommandRequest) -> MemoryArguments:
        assert request.text is not None
        tokens = request.text.tokens
        if not tokens or tokens == ("status",):
            return MemoryArguments(MemoryAction.STATUS)
        action = tokens[0].casefold()
        values = tokens[1:]
        if action == "list" and not values:
            return MemoryArguments(MemoryAction.LIST)
        if action in {"on", "off"} and not values:
            return MemoryArguments(MemoryAction.ENABLE, enabled=action == "on")
        if action == "remember" and values:
            return MemoryArguments(MemoryAction.REMEMBER, text=" ".join(values))
        if action == "forget" and len(values) == 1:
            if values[0].casefold() == "all":
                return MemoryArguments(MemoryAction.FORGET_ALL)
            try:
                item_id = UUID(values[0])
            except ValueError as exc:
                raise CommandUsageError(
                    "记忆 ID 格式不正确。",
                    include_usage=False,
                ) from exc
            return MemoryArguments(MemoryAction.FORGET, item_id=item_id)
        raise CommandUsageError(
            "",
            include_usage=True,
        )


class MemoryCommand:
    name = "memory"
    description = "查看、保存、关闭或删除自己的长期记忆。"
    category = "Agent"
    usage = (
        "memory [status|list|on|off]",
        "memory remember <内容>",
        "memory forget <ID|all>",
    )

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory
        self._errors = AgentCommandErrorPresenter()

    def definition(self) -> CommandDefinition[MemoryArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            MemoryArgumentsParser(),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=30.0),
        )

    async def handle(self, request: CommandRequest, arguments: MemoryArguments) -> None:
        person_id = request.actor.person_id
        try:
            if arguments.action is MemoryAction.STATUS:
                status = await self._memory.status(person_id)
                await request.responder.reply(
                    "长期记忆："
                    f"{'已开启' if status.enabled else '已关闭'}；"
                    f"当前 {status.item_count} 条。"
                )
            elif arguments.action is MemoryAction.LIST:
                await self._list(request, person_id)
            elif arguments.action is MemoryAction.ENABLE:
                assert arguments.enabled is not None
                await self._memory.set_enabled(person_id, arguments.enabled)
                message = (
                    "长期记忆已开启。"
                    if arguments.enabled
                    else "长期记忆已关闭；已有内容仍保留，可使用 /memory forget all 删除。"
                )
                await request.responder.reply(message)
            elif arguments.action is MemoryAction.REMEMBER:
                item = await self._memory.remember(person_id, arguments.text)
                await request.responder.reply(f"已记住。记忆 ID：{item.id}")
            elif arguments.action is MemoryAction.FORGET_ALL:
                count = await self._memory.forget_all(person_id)
                await request.responder.reply(f"已删除 {count} 条长期记忆。")
            else:
                assert arguments.item_id is not None
                deleted = await self._memory.forget(person_id, arguments.item_id)
                await request.responder.reply("已删除该记忆。" if deleted else "没有找到该记忆。")
        except MemoryDisabledError:
            await request.responder.reply("长期记忆当前已关闭；请先使用 /memory on。")
        except MemoryCapacityError:
            await request.responder.reply("长期记忆条目已满；请先删除不需要的内容。")
        except MemoryItemTooLongError:
            await request.responder.reply("这条记忆太长，请缩短后再保存。")
        except Exception as exc:
            await self._errors.reply(request, exc)

    async def _list(self, request: CommandRequest, person_id: str) -> None:
        items = await self._memory.list(person_id)
        if not items:
            await request.responder.reply("当前没有长期记忆。")
            return
        lines = ["你的长期记忆："]
        for item in items:
            text = item.content.get("text")
            if isinstance(text, str):
                display = text if len(text) <= 240 else f"{text[:237]}..."
                lines.append(f"- {item.id}: {display}")
        await request.responder.reply("\n".join(lines))
