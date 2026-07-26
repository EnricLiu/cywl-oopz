"""Owned background tasks for long-running chat replies."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from contextlib import suppress
from typing import Any

from .models import ConversationKey

logger = logging.getLogger(__name__)


class ChatTaskSupervisor:
    """Owns at most one LLM reply task per conversation and cleans every task up."""

    def __init__(self) -> None:
        self._tasks: dict[ConversationKey, asyncio.Task[object]] = {}

    def start(self, key: ConversationKey, operation: Coroutine[Any, Any, object]) -> bool:
        """Start an owned reply, or reject a duplicate request for the same session."""
        existing = self._tasks.get(key)
        if existing is not None and not existing.done():
            operation.close()
            return False

        task = asyncio.create_task(operation, name=f"chat:{key.scope}")
        self._tasks[key] = task
        task.add_done_callback(lambda completed: self._on_done(key, completed))
        return True

    def has_active(self, key: ConversationKey) -> bool:
        """Return whether the key currently owns a still-running reply task."""
        task = self._tasks.get(key)
        return task is not None and not task.done()

    async def cancel(self, key: ConversationKey) -> bool:
        """Cancel and await an active reply task so its leases and locks are released."""
        task = self._tasks.get(key)
        if task is None or task.done():
            return False
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        return True

    async def close(self) -> None:
        """Cancel and await all owned work before closing provider and database resources."""
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _on_done(self, key: ConversationKey, task: asyncio.Task[object]) -> None:
        if self._tasks.get(key) is task:
            self._tasks.pop(key, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Owned chat task failed: task=%s", task.get_name())
