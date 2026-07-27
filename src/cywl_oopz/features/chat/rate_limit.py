"""Single-process concurrency and cooldown control for LLM requests."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from cywl_oopz.core.errors import RateLimitExceeded
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.settings import ChatSettings

from .models import ConversationKey

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Counters:
    global_requests: int = 0
    channel_requests: dict[tuple[str, str], int] | None = None
    user_requests: dict[str, int] | None = None
    user_available_at: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.channel_requests = {}
        self.user_requests = {}
        self.user_available_at = {}


class RateLimitLease:
    """A release-once grant returned by ``RateLimitService.acquire``."""

    def __init__(self, service: RateLimitService, key: ConversationKey) -> None:
        self._service = service
        self._key = key
        self._released = False

    async def __aenter__(self) -> RateLimitLease:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.release()

    async def release(self) -> None:
        """Release the request slot exactly once, even in error paths."""
        if not self._released:
            self._released = True
            await self._service.release(self._key)


class RateLimitService:
    """Applies user cooldown plus user/channel/global in-flight request limits."""

    def __init__(self, settings: ChatSettings) -> None:
        self._settings = settings
        self._counters = _Counters()
        self._lock = asyncio.Lock()

    async def acquire(self, key: ConversationKey) -> RateLimitLease:
        """Fail fast when any configured limit is reached; never queue unbounded work."""
        now = time.monotonic()
        channel_key = (key.area_id, key.channel_id)
        async with self._lock:
            retry_after = max(self._counters.user_available_at.get(key.person_id, 0.0) - now, 0.0)
            if retry_after > 0.0:
                logger.warning(
                    "Chat request rejected by cooldown: conversation=%s retry_after=%.2f",
                    self._conversation_ref(key),
                    retry_after,
                )
                raise RateLimitExceeded("user cooldown", retry_after)
            if self._counters.global_requests >= self._settings.max_global_concurrency:
                logger.warning(
                    "Chat request rejected by global concurrency: conversation=%s "
                    "active=%s limit=%s",
                    self._conversation_ref(key),
                    self._counters.global_requests,
                    self._settings.max_global_concurrency,
                )
                raise RateLimitExceeded("global concurrency")
            if (
                self._counters.channel_requests.get(channel_key, 0)
                >= self._settings.max_channel_concurrency
            ):
                logger.warning(
                    "Chat request rejected by channel concurrency: conversation=%s "
                    "active=%s limit=%s",
                    self._conversation_ref(key),
                    self._counters.channel_requests.get(channel_key, 0),
                    self._settings.max_channel_concurrency,
                )
                raise RateLimitExceeded("channel concurrency")
            if (
                self._counters.user_requests.get(key.person_id, 0)
                >= self._settings.max_user_concurrency
            ):
                logger.warning(
                    "Chat request rejected by user concurrency: conversation=%s active=%s limit=%s",
                    self._conversation_ref(key),
                    self._counters.user_requests.get(key.person_id, 0),
                    self._settings.max_user_concurrency,
                )
                raise RateLimitExceeded("user concurrency")

            self._counters.global_requests += 1
            self._counters.channel_requests[channel_key] = (
                self._counters.channel_requests.get(channel_key, 0) + 1
            )
            self._counters.user_requests[key.person_id] = (
                self._counters.user_requests.get(key.person_id, 0) + 1
            )
            logger.debug(
                "Chat rate-limit lease acquired: conversation=%s global_active=%s",
                self._conversation_ref(key),
                self._counters.global_requests,
            )
        return RateLimitLease(self, key)

    async def release(self, key: ConversationKey) -> None:
        """Release counters and begin the user's cool-down period."""
        channel_key = (key.area_id, key.channel_id)
        async with self._lock:
            self._counters.global_requests -= 1
            self._decrement(self._counters.channel_requests, channel_key)
            self._decrement(self._counters.user_requests, key.person_id)
            self._counters.user_available_at[key.person_id] = (
                time.monotonic() + self._settings.user_cooldown_seconds
            )
            logger.debug(
                "Chat rate-limit lease released: conversation=%s global_active=%s",
                self._conversation_ref(key),
                self._counters.global_requests,
            )

    async def cooldown_remaining(self, key: ConversationKey) -> float:
        """Return the safe-to-display remaining user cooldown."""
        async with self._lock:
            return max(
                self._counters.user_available_at.get(key.person_id, 0.0) - time.monotonic(), 0.0
            )

    @staticmethod
    def _decrement(values: dict[object, int], key: object) -> None:
        remaining = values[key] - 1
        if remaining <= 0:
            values.pop(key, None)
        else:
            values[key] = remaining

    @staticmethod
    def _conversation_ref(key: ConversationKey) -> str:
        return opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id)
