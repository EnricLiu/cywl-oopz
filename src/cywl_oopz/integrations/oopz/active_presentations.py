"""In-process registry for Agent displays that can still edit an OOPZ message."""

from __future__ import annotations

import asyncio
from typing import Protocol


class DismissiblePresentation(Protocol):
    async def dismiss(self) -> None:
        """Permanently stop edits and terminal fallback."""


class ActivePresentationRegistry:
    """Own the one active presentation currently associated with each message."""

    def __init__(self) -> None:
        self._entries: dict[str, DismissiblePresentation] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        message_id: str,
        presentation: DismissiblePresentation,
    ) -> None:
        async with self._lock:
            self._entries[message_id] = presentation

    async def discard(
        self,
        message_id: str,
        presentation: DismissiblePresentation,
    ) -> None:
        async with self._lock:
            if self._entries.get(message_id) is presentation:
                self._entries.pop(message_id, None)

    async def dismiss(self, message_id: str) -> bool:
        async with self._lock:
            presentation = self._entries.pop(message_id, None)
        if presentation is None:
            return False
        await presentation.dismiss()
        return True
