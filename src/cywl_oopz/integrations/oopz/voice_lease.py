"""Generation-aware ownership for the single OOPZ voice backend."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum

from oopz_sdk import OopzBot

from cywl_oopz.core.observability import exception_kind, opaque_ref

logger = logging.getLogger(__name__)


class VoiceLeasePurpose(StrEnum):
    """User-facing feature currently owning OOPZ voice."""

    MUSIC = "music"
    CONVERSATION = "conversation"


class VoiceLeaseState(StrEnum):
    """Observable reservation phase without exposing the mutable token."""

    ACQUIRING = "acquiring"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class VoiceLeaseRequest:
    """Stable owner and channel requested for one exclusive lease."""

    purpose: VoiceLeasePurpose
    area_id: str
    channel_id: str
    owner_key: str

    def __post_init__(self) -> None:
        if not self.area_id.strip() or not self.channel_id.strip() or not self.owner_key.strip():
            raise ValueError("Voice lease area, channel, and owner key must not be empty")


@dataclass(frozen=True, slots=True)
class VoiceLeaseSnapshot:
    """Credential-free view used for conflict messages and health checks."""

    request: VoiceLeaseRequest
    generation: int
    state: VoiceLeaseState


class OopzVoiceLease:
    """Idempotent token whose generation prevents stale release."""

    def __init__(
        self,
        manager: OopzVoiceLeaseManager,
        request: VoiceLeaseRequest,
        generation: int,
    ) -> None:
        self._manager = manager
        self.request = request
        self.generation = generation
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> bool:
        """Leave only if this exact generation still owns the backend."""
        if self._released:
            return False
        released = await self._manager._release(self)
        self._released = True
        return released

    async def __aenter__(self) -> OopzVoiceLease:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()


class OopzVoiceLeaseManager:
    """Serialize join/leave ownership around ``OopzBot.voice``."""

    def __init__(self, bot: OopzBot) -> None:
        self._bot = bot
        self._lock = asyncio.Lock()
        self._active: OopzVoiceLease | None = None
        self._pending: VoiceLeaseSnapshot | None = None
        self._pending_done = asyncio.Event()
        self._pending_done.set()
        self._joined_without_lease = False
        self._generation = 0
        self._closed = False

    async def try_acquire(self, request: VoiceLeaseRequest) -> OopzVoiceLease | None:
        """Join and publish a lease, or return ``None`` when another owner is active."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("OOPZ voice lease manager is closed")
            if self._active is not None or self._pending is not None:
                active_purpose = (
                    self._active.request.purpose
                    if self._active is not None
                    else self._pending.request.purpose
                )
                logger.info(
                    "OOPZ voice lease unavailable: requested=%s active=%s",
                    request.purpose.value,
                    active_purpose.value,
                )
                return None

            generation = self._generation + 1
            self._generation = generation
            self._pending = VoiceLeaseSnapshot(
                request,
                generation,
                VoiceLeaseState.ACQUIRING,
            )
            self._pending_done.clear()
            logger.info(
                "Acquiring OOPZ voice lease: purpose=%s channel=%s generation=%s",
                request.purpose.value,
                self._channel_ref(request),
                generation,
            )
        try:
            await self._bot.voice.join(area=request.area_id, channel=request.channel_id)
        except BaseException:
            async with self._lock:
                if self._pending is not None and self._pending.generation == generation:
                    self._pending = None
                    self._pending_done.set()
            raise

        should_leave = False
        async with self._lock:
            if self._closed:
                should_leave = True
                self._pending = None
                self._joined_without_lease = True
            elif self._pending is None or self._pending.generation != generation:
                should_leave = True
                self._joined_without_lease = True
            else:
                lease = OopzVoiceLease(self, request, generation)
                self._active = lease
                self._pending = None
                self._pending_done.set()
        if should_leave:
            try:
                await self._bot.voice.leave()
            finally:
                self._pending_done.set()
            async with self._lock:
                self._joined_without_lease = False
            raise RuntimeError("OOPZ voice lease manager closed during join")

        logger.info(
            "OOPZ voice lease acquired: purpose=%s channel=%s generation=%s",
            request.purpose.value,
            self._channel_ref(request),
            generation,
        )
        return lease

    async def current(self) -> VoiceLeaseSnapshot | None:
        """Return the current immutable owner snapshot."""
        async with self._lock:
            if self._active is not None:
                return VoiceLeaseSnapshot(
                    self._active.request,
                    self._active.generation,
                    VoiceLeaseState.ACTIVE,
                )
            return self._pending

    async def aclose(self) -> None:
        """Prevent new owners and release the active generation if present."""
        async with self._lock:
            if (
                self._closed
                and self._pending is None
                and self._active is None
                and not self._joined_without_lease
            ):
                return
            self._closed = True
            pending = self._pending is not None
        if pending:
            await self._pending_done.wait()

        async with self._lock:
            active = self._active
            orphaned_join = self._joined_without_lease
            if active is None and not orphaned_join:
                return
            try:
                await self._bot.voice.leave()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Could not leave OOPZ voice while closing lease manager: owner=%s error=%s",
                    active.request.purpose.value if active is not None else "orphaned_join",
                    exception_kind(exc),
                )
            else:
                self._active = None
                self._joined_without_lease = False
                if active is not None:
                    active._released = True
                logger.info(
                    "OOPZ voice lease manager released active owner: purpose=%s generation=%s",
                    active.request.purpose.value if active is not None else "orphaned_join",
                    active.generation if active is not None else 0,
                )

    async def _release(self, lease: OopzVoiceLease) -> bool:
        async with self._lock:
            if self._active is not lease or self._active.generation != lease.generation:
                logger.debug(
                    "Ignored stale OOPZ voice lease release: purpose=%s generation=%s",
                    lease.request.purpose.value,
                    lease.generation,
                )
                return False
            await self._bot.voice.leave()
            self._active = None
            logger.info(
                "OOPZ voice lease released: purpose=%s channel=%s generation=%s",
                lease.request.purpose.value,
                self._channel_ref(lease.request),
                lease.generation,
            )
            return True

    @staticmethod
    def _channel_ref(request: VoiceLeaseRequest) -> str:
        return opaque_ref(request.area_id, request.channel_id)
