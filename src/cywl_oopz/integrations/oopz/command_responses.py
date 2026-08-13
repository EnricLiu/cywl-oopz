"""OOPZ implementation of the project command response port."""

from __future__ import annotations

from typing import Any


class OopzCommandResponder:
    """Delegate to the tracked OOPZ context without exposing it to new handlers."""

    def __init__(self, context: Any) -> None:
        self._context = context

    async def reply(self, text: str) -> Any:
        return await self._context.reply(text)

    async def send(self, text: str) -> Any:
        return await self._context.send(text)

    async def react(self, emoji: str) -> Any:
        return await self._context.react(emoji)
