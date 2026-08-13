"""Small built-in commands used to validate bot connectivity."""

from __future__ import annotations

from dataclasses import dataclass

from oopz_sdk.events.context import EventContext

from cywl_oopz.core.health import HealthRegistry, HealthState

from .catalog import CommandSpec
from .definitions import (
    CommandDefinition,
    CommandUsageError,
    NoArguments,
    NoArgumentsParser,
    PublicCommandAuthorization,
)
from .models import CommandRequest
from .responses import CommandMessage, CommandMessageBudget, MessageOverflowPolicy
from .router import CommandRouter, ParsedCommand


class PingCommand:
    """Reply with a short liveness response."""

    name = "ping"
    description = "检查机器人是否在线。"
    category = "基础"
    usage = ("ping",)

    def definition(self) -> CommandDefinition[NoArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            NoArgumentsParser(),
            self,
            PublicCommandAuthorization(),
        )

    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del arguments
        await request.responder.reply("pong")

    async def execute(self, _: ParsedCommand, context: EventContext) -> None:
        await context.reply("pong")


class HelpCommand:
    """Render the commands currently registered in the router."""

    name = "help"
    description = "显示可用命令。"
    category = "基础"
    usage = ("help", "help <命令>")
    examples = ("help music", "help role")

    def __init__(
        self,
        router: CommandRouter,
        budget: CommandMessageBudget | None = None,
    ) -> None:
        self._router = router
        self._budget = budget or CommandMessageBudget()

    def definition(self) -> CommandDefinition[HelpArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            HelpArgumentsParser(),
            self,
            PublicCommandAuthorization(),
        )

    async def handle(self, request: CommandRequest, arguments: HelpArguments) -> None:
        specs = await self._router.available_specs_for(request)
        if arguments.command_name:
            spec = next(
                (item for item in specs if item.matches(arguments.command_name)),
                None,
            )
            if spec is None:
                await request.responder.reply(
                    f"没有找到命令：{arguments.command_name}\n"
                    f"输入 {self._router.prefix}help 查看可用命令。"
                )
                return
            await request.responder.reply(self._detail(spec))
            return
        await request.responder.reply(self._overview(specs))

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        if len(command.arguments) > 1:
            await context.reply(f"用法：{self._router.prefix}help [命令]")
            return
        specs = await self._router.available_specs(context)
        if command.arguments:
            requested = command.arguments[0]
            spec = next((item for item in specs if item.matches(requested)), None)
            if spec is None:
                await context.reply(
                    f"没有找到命令：{requested}\n输入 {self._router.prefix}help 查看可用命令。"
                )
                return
            await self._reply_pages(context, self._detail(spec))
            return
        await self._reply_pages(context, self._overview(specs))

    def _overview(self, specs: tuple[CommandSpec, ...]) -> str:
        grouped: dict[str, list[CommandSpec]] = {}
        for spec in specs:
            grouped.setdefault(spec.category, []).append(spec)
        lines = ["**可用命令**"]
        for category in sorted(grouped):
            lines.extend(("", f"**{category}**"))
            lines.extend(
                f"{self._router.prefix}{spec.name} — {spec.summary}" for spec in grouped[category]
            )
        lines.extend(
            (
                "",
                f"输入 {self._router.prefix}help <命令> 查看详细用法。",
            )
        )
        return "\n".join(lines)

    def _detail(self, spec: CommandSpec) -> str:
        lines = [f"**{self._router.prefix}{spec.name}**", spec.summary, "", "**用法**"]
        lines.extend(f"{self._router.prefix}{usage}" for usage in spec.usage)
        if spec.aliases:
            aliases = "、".join(f"{self._router.prefix}{alias}" for alias in spec.aliases)
            lines.extend(("", f"别名：{aliases}"))
        if spec.examples:
            lines.extend(("", "**示例**"))
            lines.extend(f"{self._router.prefix}{example}" for example in spec.examples)
        return "\n".join(lines)

    async def _reply_pages(self, context: EventContext, text: str) -> None:
        message = CommandMessage(text, MessageOverflowPolicy.PAGINATE)
        for page in self._budget.pages(message):
            await context.reply(page)


@dataclass(frozen=True, slots=True)
class HelpArguments:
    command_name: str = ""


class HelpArgumentsParser:
    def parse(self, request: CommandRequest) -> HelpArguments:
        assert request.text is not None
        if len(request.text.tokens) > 1:
            raise CommandUsageError("")
        return HelpArguments(request.text.tokens[0] if request.text.tokens else "")


class StatusCommand:
    """Show safe in-process component health without exposing configuration."""

    name = "status"
    description = "查看机器人组件状态。"
    category = "基础"
    usage = ("status",)

    def __init__(self, health: HealthRegistry) -> None:
        self._health = health

    def definition(self) -> CommandDefinition[NoArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            NoArgumentsParser(),
            self,
            PublicCommandAuthorization(),
        )

    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del arguments
        await request.responder.reply(self._render())

    async def execute(self, _: ParsedCommand, context: EventContext) -> None:
        await context.reply(self._render())

    def _render(self) -> str:
        icons = {
            HealthState.HEALTHY: "正常",
            HealthState.PENDING: "检查中",
            HealthState.DEGRADED: "异常",
            HealthState.DISABLED: "已禁用",
        }
        checks = self._health.snapshot()
        if not checks:
            return "状态：尚未初始化。"
        lines = ["组件状态："]
        lines.extend(f"{check.name}：{icons[check.state]}" for check in checks)
        return "\n".join(lines)
