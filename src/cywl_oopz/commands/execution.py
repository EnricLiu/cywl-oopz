"""Application-owned lifecycle for background command executions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class CommandTaskSupervisor:
    """Own every top-level background command until completion or shutdown."""

    def __init__(self, *, drain_timeout_seconds: float = 2.0) -> None:
        if drain_timeout_seconds < 0:
            raise ValueError("Command drain timeout must not be negative")
        self._drain_timeout_seconds = drain_timeout_seconds
        self._tasks: set[asyncio.Task[object]] = set()
        self._accepting = True

    @property
    def accepting(self) -> bool:
        """Return whether new background commands may be accepted."""
        return self._accepting

    @property
    def active_count(self) -> int:
        """Return the number of command tasks still owned by the supervisor."""
        return sum(not task.done() for task in self._tasks)

    def start(
        self,
        command_name: str,
        request_ref: str,
        operation: Coroutine[Any, Any, object],
    ) -> bool:
        """Start owned work, or close the rejected coroutine during shutdown."""
        if not self._accepting:
            operation.close()
            logger.debug(
                "Rejected background command during shutdown: command=%s request=%s",
                command_name,
                request_ref,
            )
            return False
        task = asyncio.create_task(
            operation,
            name=f"command:{command_name}:{request_ref}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        logger.debug(
            "Background command task started: command=%s request=%s active=%s",
            command_name,
            request_ref,
            self.active_count,
        )
        return True

    async def close(self) -> None:
        """Stop admission, briefly drain commands, then cancel and await leftovers."""
        self._accepting = False
        tasks = set(self._tasks)
        pending = {task for task in tasks if not task.done()}
        if not tasks:
            self._tasks.clear()
            return
        logger.info(
            "Draining background commands: count=%s timeout_seconds=%.1f",
            len(pending),
            self._drain_timeout_seconds,
        )
        if self._drain_timeout_seconds:
            _, pending = await asyncio.wait(
                pending,
                timeout=self._drain_timeout_seconds,
            )
        if pending:
            logger.info("Cancelling background commands after drain: count=%s", len(pending))
            for task in pending:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _on_done(self, task: asyncio.Task[object]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            logger.debug("Background command task cancelled: task=%s", task.get_name())
            return
        try:
            task.result()
        except Exception as exc:
            # Router error boundaries consume handler failures. Reaching this branch
            # still retrieves the exception so asyncio never reports an orphan task.
            logger.error(
                "Background command task boundary failed: task=%s error=%s",
                task.get_name(),
                type(exc).__name__,
            )
