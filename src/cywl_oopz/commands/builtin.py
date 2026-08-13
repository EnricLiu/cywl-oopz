"""Small built-in commands used to validate bot connectivity."""

from __future__ import annotations

from oopz_sdk.events.context import EventContext

from cywl_oopz.core.health import HealthRegistry, HealthState

from .router import CommandRouter, ParsedCommand


class PingCommand:
    """Reply with a short liveness response."""

    name = "ping"
    description = "检查机器人是否在线。"

    async def execute(self, _: ParsedCommand, context: EventContext) -> None:
        await context.reply("pong")


class HelpCommand:
    """Render the commands currently registered in the router."""

    name = "help"
    description = "显示可用命令。"

    def __init__(self, router: CommandRouter) -> None:
        self._router = router

    async def execute(self, _: ParsedCommand, context: EventContext) -> None:
        lines = ["可用命令："]
        lines.extend(
            f"{self._router.prefix}{command.name} — {command.description}"
            for command in await self._router.available_commands(context)
        )
        await context.reply("\n".join(lines))


class StatusCommand:
    """Show safe in-process component health without exposing configuration."""

    name = "status"
    description = "查看机器人组件状态。"

    def __init__(self, health: HealthRegistry) -> None:
        self._health = health

    async def execute(self, _: ParsedCommand, context: EventContext) -> None:
        icons = {
            HealthState.HEALTHY: "正常",
            HealthState.PENDING: "检查中",
            HealthState.DEGRADED: "异常",
            HealthState.DISABLED: "已禁用",
        }
        checks = self._health.snapshot()
        if not checks:
            await context.reply("状态：尚未初始化。")
            return
        lines = ["组件状态："]
        lines.extend(f"{check.name}：{icons[check.state]}" for check in checks)
        await context.reply("\n".join(lines))
