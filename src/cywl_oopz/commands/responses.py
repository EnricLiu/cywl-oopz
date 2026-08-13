"""Project-owned response port for command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class MessageOverflowPolicy(StrEnum):
    """Explicit behavior when one response exceeds the platform budget."""

    PAGINATE = "paginate"
    TRUNCATE = "truncate"
    REJECT = "reject"


class CommandMessageTooLongError(ValueError):
    """Raised when an atomic response cannot be emitted within its budget."""


@dataclass(frozen=True, slots=True)
class CommandMessage:
    """One semantic response and its chosen overflow behavior."""

    text: str
    overflow: MessageOverflowPolicy = MessageOverflowPolicy.PAGINATE


class CommandMessageBudget:
    """Bound messages by OOPZ-compatible UTF-16 code units."""

    def __init__(self, max_units: int = 1950) -> None:
        if max_units < 16:
            raise ValueError("Command message budget must be at least 16 units")
        self.max_units = max_units

    @staticmethod
    def units(text: str) -> int:
        return len(text.encode("utf-16-le")) // 2

    def pages(self, message: str | CommandMessage) -> tuple[str, ...]:
        item = message if isinstance(message, CommandMessage) else CommandMessage(message)
        if self.units(item.text) <= self.max_units:
            return (item.text,)
        if item.overflow is MessageOverflowPolicy.REJECT:
            raise CommandMessageTooLongError(
                f"Command response exceeds {self.max_units} UTF-16 units"
            )
        if item.overflow is MessageOverflowPolicy.TRUNCATE:
            return (self._truncate(item.text),)
        return self._paginate(item.text)

    def _truncate(self, text: str) -> str:
        suffix = "…"
        return f"{self._take(text, self.max_units - self.units(suffix))[0]}{suffix}"

    def _paginate(self, text: str) -> tuple[str, ...]:
        pages: list[str] = []
        remaining = text
        while remaining:
            page, remaining = self._take(remaining, self.max_units)
            if remaining and "\n" in page:
                boundary = page.rfind("\n")
                remainder_prefix = page[boundary + 1 :]
                if remainder_prefix:
                    remaining = f"{remainder_prefix}{remaining}"
                page = page[:boundary]
            if not page:
                page, remaining = self._take(remaining, self.max_units)
            pages.append(page)
            remaining = remaining.lstrip("\n")
        return tuple(pages)

    @staticmethod
    def _take(text: str, limit: int) -> tuple[str, str]:
        units = 0
        index = 0
        for index, character in enumerate(text):
            character_units = 2 if ord(character) > 0xFFFF else 1
            if units + character_units > limit:
                return text[:index], text[index:]
            units += character_units
        return text, ""


class CommandResponder(Protocol):
    """Deliver command output without exposing a platform event context."""

    async def reply(self, message: str | CommandMessage) -> Any:
        """Reply to the source invocation."""

    async def send(self, message: str | CommandMessage) -> Any:
        """Send a message beside the source invocation."""

    async def react(self, emoji: str) -> Any:
        """Add one reaction to the source message."""
