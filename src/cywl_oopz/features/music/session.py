"""Music queue transitions and per-channel ownership state."""

from __future__ import annotations

import asyncio
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from .models import (
    EnqueueResult,
    MusicFailure,
    MusicFailureCode,
    MusicFailureScope,
    MusicPlaybackPolicy,
    PlayableTrack,
    PlaybackOrder,
    PlaybackState,
    QueuedTrack,
    RepeatPolicy,
)
from .ports import MusicPlayback


@dataclass(slots=True)
class MusicQueueState:
    """Queue values and transitions; callers serialize access with the session lock."""

    queue: deque[QueuedTrack] = field(default_factory=deque)
    cycle_history: deque[QueuedTrack] = field(default_factory=deque)
    current: QueuedTrack | None = None
    state: PlaybackState = PlaybackState.IDLE
    policy: MusicPlaybackPolicy = field(default_factory=MusicPlaybackPolicy)
    revision: int = 0
    idempotent_enqueues: OrderedDict[str, EnqueueResult] = field(default_factory=OrderedDict)
    last_failure: MusicFailure | None = None

    def take_next(self, rng: random.Random) -> QueuedTrack:
        if self.policy.order is PlaybackOrder.SHUFFLE and len(self.queue) > 1:
            index = rng.randrange(len(self.queue))
            self.queue.rotate(-index)
            item = self.queue.popleft()
            self.queue.rotate(index)
            return item
        return self.queue.popleft()

    def retain_completed(self, item: QueuedTrack) -> None:
        if self.policy.repeat is RepeatPolicy.ONE:
            self.queue.appendleft(item)
        else:
            self.cycle_history.append(item)

    def record_failure(
        self,
        item: QueuedTrack | None,
        code: MusicFailureCode,
        scope: MusicFailureScope,
        *,
        recoverable: bool,
        retry_count: int,
    ) -> None:
        self.last_failure = MusicFailure(
            code,
            scope,
            recoverable,
            item.id if item is not None else None,
            retry_count,
        )


@dataclass(slots=True)
class MusicSession(MusicQueueState):
    """Per-channel task and lease ownership around the queue state."""

    playback: MusicPlayback | None = None
    resolve_task: asyncio.Task[PlayableTrack] | None = None
    voice_reserved: bool = False
    worker: asyncio.Task[None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    skip_requested: asyncio.Event = field(default_factory=asyncio.Event)
    retain_skipped_for_cycle: bool = False

    def cancel_resolve(self) -> None:
        """Cancel the current lookup while the caller holds the session lock."""
        if self.resolve_task is not None and not self.resolve_task.done():
            self.resolve_task.cancel()
