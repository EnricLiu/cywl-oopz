"""Owned asyncio task lifecycle shared by long-running bot features."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine, Hashable
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


class TaskSupervisor[KeyT: Hashable]:
    """Own at most one task per key and await every task during cancellation."""

    def __init__(self, task_name: Callable[[KeyT], str]) -> None:
        self._task_name = task_name
        self._tasks: dict[KeyT, asyncio.Task[object]] = {}

    def start(self, key: KeyT, operation: Coroutine[Any, Any, object]) -> bool:
        """Start owned work, or close and reject a duplicate coroutine."""
        existing = self._tasks.get(key)
        if existing is not None and not existing.done():
            operation.close()
            return False

        task = asyncio.create_task(operation, name=self._task_name(key))
        self._tasks[key] = task
        task.add_done_callback(lambda completed: self._on_done(key, completed))
        return True

    def has_active(self, key: KeyT) -> bool:
        """Return whether this key has a still-running task."""
        task = self._tasks.get(key)
        return task is not None and not task.done()

    async def cancel(self, key: KeyT) -> bool:
        """Cancel and await active work so leases and locks are released."""
        task = self._tasks.get(key)
        if task is None or task.done():
            return False
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        return True

    async def close(self) -> None:
        """Cancel and await all work before application resources close."""
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _on_done(self, key: KeyT, task: asyncio.Task[object]) -> None:
        if self._tasks.get(key) is task:
            self._tasks.pop(key, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Owned task failed: task=%s", task.get_name())
