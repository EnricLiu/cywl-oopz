"""Project-owned response port for command handlers."""

from __future__ import annotations

from typing import Any, Protocol


class CommandResponder(Protocol):
    """Deliver command output without exposing a platform event context."""

    async def reply(self, text: str) -> Any:
        """Reply to the source invocation."""

    async def send(self, text: str) -> Any:
        """Send a message beside the source invocation."""

    async def react(self, emoji: str) -> Any:
        """Add one reaction to the source message."""
