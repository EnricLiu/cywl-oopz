"""Shared master mixer bus independent from OOPZ and feature-specific PCM models."""

from __future__ import annotations

import asyncio

from .errors import AudioBusFailedError, AudioLedgerError
from .ledger import MasterPlayoutLedger, RemixPlan, SourceKey
from .mixer import AudioMixer, DuckingSnapshot
from .models import AUDIO_BLOCK_FRAMES, AudioBlock, SourcePlaybackCursor
from .ports import MasterPcmOutput


class SharedAudioMixerBus:
    """Own the only master writer and map its clock back to source cursors."""

    def __init__(
        self,
        master: MasterPcmOutput,
        *,
        max_buffer_frames: int,
        mixer: AudioMixer | None = None,
    ) -> None:
        if max_buffer_frames < AUDIO_BLOCK_FRAMES:
            raise ValueError("Shared audio bus must retain at least one block")
        self._master = master
        self._mixer = mixer or AudioMixer()
        self._ledger = MasterPlayoutLedger(max_buffer_frames=max_buffer_frames)
        self._operation_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._failed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failed(self) -> bool:
        return self._failed

    async def write_voice(
        self,
        blocks: tuple[AudioBlock, ...],
    ) -> dict[SourceKey, SourcePlaybackCursor]:
        """Mix and synchronously submit ordered VOICE blocks to the master."""
        async with self._operation_lock:
            self._require_healthy()
            cursors = self._ledger.source_cursors()
            for block in blocks:
                mixed = self._mixer.mix(
                    None,
                    block,
                    DuckingSnapshot(conversation_active=True),
                )
                entry_id = self._ledger.register_pending(mixed)
                try:
                    master_cursor = await self._master.write(mixed.pcm_s16le)
                    cursors = self._ledger.mark_accepted(entry_id, master_cursor)
                except BaseException:
                    self._failed = True
                    raise
            return cursors

    async def flush_voice(self, discard: frozenset[SourceKey]) -> RemixPlan:
        """Flush one master epoch and discard selected VOICE generations."""
        async with self._operation_lock:
            self._require_healthy()
            try:
                old_cursor = await self._master.flush()
                plan = self._ledger.remix(old_cursor, discard=discard)
            except BaseException:
                self._failed = True
                raise
            if plan.survivors:
                self._failed = True
                raise AudioLedgerError("Voice-only bus cannot replay another active source")
            self._mixer.reset_dynamics()
            return plan

    async def drain(self) -> dict[SourceKey, SourcePlaybackCursor]:
        """Wait until the master renderer reports all submitted audio rendered."""
        async with self._operation_lock:
            self._require_healthy()
            try:
                cursor = await self._master.drain()
                return self._ledger.observe(cursor)
            except BaseException:
                self._failed = True
                raise

    async def observe(self) -> dict[SourceKey, SourcePlaybackCursor]:
        """Reconcile a non-blocking master stats snapshot."""
        async with self._operation_lock:
            self._require_healthy()
            try:
                return self._ledger.observe(self._master.cursor)
            except BaseException:
                self._failed = True
                raise

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            await self._master.aclose()
            self._closed = True

    def _require_healthy(self) -> None:
        if self._closed:
            raise AudioBusFailedError("Shared audio bus is closed")
        if self._failed:
            raise AudioBusFailedError("Shared audio bus transport has failed")


def master_buffer_frames(max_buffer_ms: int) -> int:
    """Convert a positive canonical master window to exact frame capacity."""
    if max_buffer_ms <= 0:
        raise ValueError("Master buffer duration must be positive")
    return 48_000 * max_buffer_ms // 1_000
