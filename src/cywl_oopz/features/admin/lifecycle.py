"""Application-level graceful shutdown and restart coordination."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from .models import ShutdownDisposition

logger = logging.getLogger(__name__)

RestartConfirmation = Callable[[], Awaitable[object]]


class ApplicationLifecycleCoordinator:
    """Commit one restart request only after its confirmation was delivered."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._disposition = ShutdownDisposition.NORMAL

    @property
    def disposition(self) -> ShutdownDisposition:
        return self._disposition

    @property
    def restart_requested(self) -> bool:
        return self._disposition is ShutdownDisposition.RESTART

    async def request_restart(
        self,
        actor_ref: str,
        confirm: RestartConfirmation,
    ) -> bool:
        """Serialize requests, send confirmation, then commit first-writer-wins state."""
        async with self._lock:
            if self.restart_requested:
                return False
            await confirm()
            self._disposition = ShutdownDisposition.RESTART
            self._event.set()
            logger.info("Application restart requested: actor=%s", actor_ref)
            return True

    async def wait(self) -> ShutdownDisposition:
        """Wait until an explicit non-normal shutdown disposition is committed."""
        await self._event.wait()
        return self._disposition
