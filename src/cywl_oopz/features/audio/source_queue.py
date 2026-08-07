"""Bounded generation-aware queues for canonical source blocks."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, replace
from uuid import UUID

from .errors import AudioBackpressureError, AudioQueueClosedError
from .models import AUDIO_BLOCK_DURATION_MS, AudioBlock, AudioSourceKind


@dataclass(frozen=True, slots=True)
class AudioSourceQueueStats:
    """Cumulative source queue and explicit discard counters."""

    enqueued_blocks: int = 0
    delivered_blocks: int = 0
    flushed_blocks: int = 0
    stale_blocks: int = 0
    backpressure_timeouts: int = 0


class AudioSourceQueue:
    """Preserve source audio until explicit generation flush or bounded failure."""

    def __init__(
        self,
        source_id: UUID,
        kind: AudioSourceKind,
        *,
        max_duration_ms: int,
        put_timeout_seconds: float = 1.0,
    ) -> None:
        if max_duration_ms < AUDIO_BLOCK_DURATION_MS or put_timeout_seconds <= 0:
            raise ValueError("Audio source queue bounds must be positive")
        self.source_id = source_id
        self.kind = kind
        self._max_blocks = max_duration_ms // AUDIO_BLOCK_DURATION_MS
        self._put_timeout_seconds = put_timeout_seconds
        self._generation = 0
        self._items: deque[AudioBlock] = deque()
        self._condition = asyncio.Condition()
        self._closed = False
        self._stats = AudioSourceQueueStats()

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def qsize(self) -> int:
        return len(self._items)

    @property
    def max_blocks(self) -> int:
        return self._max_blocks

    @property
    def buffered_duration_ms(self) -> int:
        return len(self._items) * AUDIO_BLOCK_DURATION_MS

    @property
    def stats(self) -> AudioSourceQueueStats:
        return self._stats

    async def start_generation(self, generation: int) -> int:
        """Switch to a newer source generation and discard all queued old audio."""
        async with self._condition:
            if self._closed:
                raise AudioQueueClosedError
            if generation <= self._generation:
                raise ValueError("Audio source generation must increase")
            flushed = len(self._items)
            self._items.clear()
            self._generation = generation
            self._stats = replace(
                self._stats,
                flushed_blocks=self._stats.flushed_blocks + flushed,
            )
            self._condition.notify_all()
            return flushed

    async def put(self, block: AudioBlock) -> bool:
        """Wait for bounded capacity, returning false for an invalidated generation."""
        self._validate_block(block)
        try:
            async with asyncio.timeout(self._put_timeout_seconds):
                async with self._condition:
                    while True:
                        if self._closed:
                            raise AudioQueueClosedError
                        if block.generation != self._generation:
                            self._stats = replace(
                                self._stats,
                                stale_blocks=self._stats.stale_blocks + 1,
                            )
                            return False
                        if len(self._items) < self._max_blocks:
                            self._items.append(block)
                            self._stats = replace(
                                self._stats,
                                enqueued_blocks=self._stats.enqueued_blocks + 1,
                            )
                            self._condition.notify_all()
                            return True
                        await self._condition.wait()
        except TimeoutError as exc:
            async with self._condition:
                self._stats = replace(
                    self._stats,
                    backpressure_timeouts=self._stats.backpressure_timeouts + 1,
                )
            raise AudioBackpressureError("Audio source queue wait timed out") from exc

    async def get(self) -> AudioBlock:
        async with self._condition:
            while not self._items:
                if self._closed:
                    raise AudioQueueClosedError
                await self._condition.wait()
            block = self._items.popleft()
            self._stats = replace(
                self._stats,
                delivered_blocks=self._stats.delivered_blocks + 1,
            )
            self._condition.notify_all()
            return block

    async def wait_empty(self, generation: int) -> None:
        async with self._condition:
            while (
                not self._closed
                and generation == self._generation
                and any(item.generation == generation for item in self._items)
            ):
                await self._condition.wait()

    async def aclose(self) -> None:
        async with self._condition:
            if self._closed:
                return
            self._closed = True
            flushed = len(self._items)
            self._items.clear()
            self._stats = replace(
                self._stats,
                flushed_blocks=self._stats.flushed_blocks + flushed,
            )
            self._condition.notify_all()

    def _validate_block(self, block: AudioBlock) -> None:
        if block.source_id != self.source_id or block.kind is not self.kind:
            raise ValueError("Audio block belongs to a different source queue")
