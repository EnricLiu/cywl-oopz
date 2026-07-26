"""Per-conversation lock ownership with bounded idle lock retention."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .models import ConversationKey


@dataclass(slots=True)
class _LockSlot:
    lock: asyncio.Lock
    users: int = 0


class ConversationLockPool:
    """Serialises one conversation while allowing unrelated keys to run concurrently."""

    def __init__(self) -> None:
        self._slots: dict[ConversationKey, _LockSlot] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, key: ConversationKey):
        """Yield after acquiring a key-specific lock and remove it when idle."""
        async with self._guard:
            slot = self._slots.setdefault(key, _LockSlot(lock=asyncio.Lock()))
            slot.users += 1
        try:
            await slot.lock.acquire()
        except BaseException:
            await self._release_reference(key, slot)
            raise

        try:
            yield
        finally:
            slot.lock.release()
            await self._release_reference(key, slot)

    async def _release_reference(self, key: ConversationKey, slot: _LockSlot) -> None:
        async with self._guard:
            slot.users -= 1
            if slot.users == 0 and not slot.lock.locked():
                self._slots.pop(key, None)
