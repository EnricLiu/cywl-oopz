"""OOPZ implementation of the project command response port."""

from __future__ import annotations

from typing import Any

from cywl_oopz.commands.responses import CommandMessage, CommandMessageBudget


class OopzCommandResponder:
    """Delegate to the tracked OOPZ context without exposing it to new handlers."""

    def __init__(
        self,
        context: Any,
        budget: CommandMessageBudget | None = None,
    ) -> None:
        self._context = context
        self._budget = budget or CommandMessageBudget()

    async def reply(self, message: str | CommandMessage) -> Any:
        return await self._emit(self._context.reply, message)

    async def send(self, message: str | CommandMessage) -> Any:
        return await self._emit(self._context.send, message)

    async def react(self, emoji: str) -> Any:
        return await self._context.react(emoji)

    async def _emit(self, operation: Any, message: str | CommandMessage) -> Any:
        results = tuple([await operation(page) for page in self._budget.pages(message)])
        return results[0] if len(results) == 1 else results
