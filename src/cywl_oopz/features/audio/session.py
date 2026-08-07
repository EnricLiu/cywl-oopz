"""Shared master mixer bus independent from OOPZ and feature-specific PCM models."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .dynamics import float32_stereo_to_s16le
from .errors import AudioBackpressureError, AudioBusFailedError, AudioLedgerError
from .ledger import MasterPlayoutLedger, RemixPlan, ReplaySegment, SourceKey
from .mixer import AudioMixer, DuckingSnapshot, MixedAudioBlock
from .models import (
    AUDIO_BLOCK_DURATION_MS,
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AudioBlock,
    AudioMixerBusStats,
    AudioSourceKind,
    DuckingReason,
    SourcePlaybackCursor,
    SourceSlice,
    VoiceParticipantKind,
)
from .ports import MasterPcmOutput

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _QueuedBlock:
    block: AudioBlock
    accepted: asyncio.Future[dict[SourceKey, SourcePlaybackCursor]]


class SharedAudioMixerBus:
    """Own one master writer, two bounded lanes, and selective remix barriers."""

    def __init__(
        self,
        master: MasterPcmOutput,
        *,
        max_buffer_frames: int,
        music_queue_ms: int = 500,
        voice_queue_ms: int = 60,
        master_target_buffer_ms: int = 60,
        put_timeout_seconds: float = 1.0,
        mixer: AudioMixer | None = None,
    ) -> None:
        if max_buffer_frames < AUDIO_BLOCK_FRAMES:
            raise ValueError("Shared audio bus must retain at least one block")
        if min(music_queue_ms, voice_queue_ms) < AUDIO_BLOCK_DURATION_MS:
            raise ValueError("Shared audio source queues must retain at least one block")
        if master_target_buffer_ms < AUDIO_BLOCK_DURATION_MS:
            raise ValueError("Master target buffer must retain at least one block")
        if put_timeout_seconds <= 0:
            raise ValueError("Shared audio queue timeout must be positive")
        self._master = master
        self._mixer = mixer or AudioMixer()
        self._ledger = MasterPlayoutLedger(max_buffer_frames=max_buffer_frames)
        self._queue_limits = {
            AudioSourceKind.MUSIC: music_queue_ms // AUDIO_BLOCK_DURATION_MS,
            AudioSourceKind.VOICE: voice_queue_ms // AUDIO_BLOCK_DURATION_MS,
        }
        self._put_timeout_seconds = put_timeout_seconds
        self._target_buffer_frames = master_buffer_frames(master_target_buffer_ms)
        self._queues: dict[AudioSourceKind, deque[_QueuedBlock]] = {
            AudioSourceKind.MUSIC: deque(),
            AudioSourceKind.VOICE: deque(),
        }
        self._condition = asyncio.Condition()
        self._operation_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._participants: frozenset[VoiceParticipantKind] = frozenset()
        self._ducking_reasons: frozenset[DuckingReason] = frozenset()
        self._closing = False
        self._closed = False
        self._failed = False
        self._master_cursor_observed_at = time.monotonic()
        self._master_started = False
        self._master_max_buffered_frames = 0
        self._master_underrun_count = 0
        self._mixer_deadline_miss_count = 0
        self._remix_count = 0
        self._last_remix_ms = 0.0
        self._max_remix_ms = 0.0
        self._replayed_frames = {
            AudioSourceKind.MUSIC: 0,
            AudioSourceKind.VOICE: 0,
        }
        self._limiter_active_blocks = 0
        self._max_gain_reduction_db = 0.0
        self._hard_clip_samples = 0
        self._writer = asyncio.create_task(self._writer_loop(), name="shared-audio-writer")

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def failed(self) -> bool:
        return self._failed

    def update_participants(self, participants: frozenset[VoiceParticipantKind]) -> None:
        """Publish participant presence on the owning event loop."""
        self._participants = participants
        if VoiceParticipantKind.CONVERSATION not in participants:
            self._ducking_reasons = frozenset()

    async def set_user_speaking(self, speaking: bool) -> None:
        """Own the USER_SPEECH ducking reason independently from VOICE playout."""
        await self._set_ducking_reason(DuckingReason.USER_SPEECH, speaking)

    async def set_voice_playout(self, playing: bool) -> None:
        """Own the VOICE_PLAYOUT reason for the active response tail."""
        await self._set_ducking_reason(DuckingReason.VOICE_PLAYOUT, playing)

    async def _set_ducking_reason(self, reason: DuckingReason, active: bool) -> None:
        reasons = set(self._ducking_reasons)
        if active:
            reasons.add(reason)
        else:
            reasons.discard(reason)
        updated = frozenset(reasons)
        if updated != self._ducking_reasons:
            self._ducking_reasons = updated
            logger.debug(
                "Updated audio ducking reasons: active=%s reasons=%s",
                active,
                ",".join(sorted(item.value for item in updated)) or "none",
            )

    async def write_voice(
        self,
        blocks: tuple[AudioBlock, ...],
    ) -> dict[SourceKey, SourcePlaybackCursor]:
        return await self._write_blocks(blocks, AudioSourceKind.VOICE)

    async def write_music(
        self,
        blocks: tuple[AudioBlock, ...],
    ) -> dict[SourceKey, SourcePlaybackCursor]:
        return await self._write_blocks(blocks, AudioSourceKind.MUSIC)

    async def _write_blocks(
        self,
        blocks: tuple[AudioBlock, ...],
        kind: AudioSourceKind,
    ) -> dict[SourceKey, SourcePlaybackCursor]:
        cursors = self._ledger.source_cursors()
        if kind is AudioSourceKind.VOICE and blocks:
            await self.set_voice_playout(True)
        for block in blocks:
            if block.kind is not kind:
                raise ValueError("Audio block was submitted to the wrong bus lane")
            waiter = asyncio.get_running_loop().create_future()
            queued = _QueuedBlock(block, waiter)
            try:
                async with asyncio.timeout(self._put_timeout_seconds):
                    async with self._condition:
                        while len(self._queues[kind]) >= self._queue_limits[kind]:
                            self._require_healthy()
                            await self._condition.wait()
                        self._require_healthy()
                        self._queues[kind].append(queued)
                        self._condition.notify_all()
            except TimeoutError as exc:
                raise AudioBackpressureError("Shared audio source queue wait timed out") from exc
            cursors = await asyncio.shield(waiter)
        return cursors

    async def flush_voice(self, discard: frozenset[SourceKey]) -> RemixPlan:
        plan = await self.flush_source(discard)
        await self.set_voice_playout(False)
        return plan

    async def flush_source(self, discard: frozenset[SourceKey]) -> RemixPlan:
        """Discard one lane generation and replay the other lane's unrendered suffix."""
        started_at = time.monotonic()
        async with self._operation_lock:
            self._require_healthy()
            removed = await self._discard_queued(discard)
            try:
                old_cursor = await self._master.flush()
                self._master_cursor_observed_at = time.monotonic()
                plan = self._ledger.remix(old_cursor, discard=discard)
                self._mixer.reset_dynamics()
                self._master_started = False
                await self._submit_replays(plan)
                self._ledger.forget_sources(discard)
            except BaseException:
                self._failed = True
                self._fail_waiters(removed, AudioBusFailedError("Audio remix barrier failed"))
                raise
            for item in removed:
                if not item.accepted.done():
                    item.accepted.set_result(plan.source_cursors)
            elapsed_ms = (time.monotonic() - started_at) * 1_000
            self._record_remix(plan, elapsed_ms)
            logger.info(
                "Remixed shared audio master: discarded=%s replay_segments=%s elapsed_ms=%.1f",
                len(discard),
                len(plan.survivors),
                elapsed_ms,
            )
            return plan

    async def drain(
        self,
        source: SourceKey | None = None,
        *,
        release_source: bool = False,
    ) -> dict[SourceKey, SourcePlaybackCursor]:
        """Drain current master audio, returning source-local rendered cursors."""
        if release_source and source is None:
            raise ValueError("Releasing a drained source requires its key")
        if source is not None and self._other_participant_active(source.kind):
            deadline = time.monotonic() + max(
                0.1,
                self._target_buffer_frames / AUDIO_SAMPLE_RATE + 0.1,
            )
            while time.monotonic() < deadline:
                async with self._operation_lock:
                    self._require_healthy()
                    observed = self._ledger.observe(self._master.cursor)
                    cursor = observed.get(source)
                    if cursor is None or cursor.rendered_frames >= cursor.accepted_frames:
                        if release_source:
                            self._ledger.forget_sources(frozenset({source}))
                        return observed
                await asyncio.sleep(AUDIO_BLOCK_DURATION_MS / 2_000)
        async with self._operation_lock:
            self._require_healthy()
            try:
                observed = self._ledger.observe(self._master.cursor)
                if source is not None:
                    cursor = observed.get(source)
                    if cursor is not None and cursor.rendered_frames >= cursor.accepted_frames:
                        if release_source:
                            self._ledger.forget_sources(frozenset({source}))
                        return observed
                cursor = await self._master.drain()
                self._master_cursor_observed_at = time.monotonic()
                observed = self._ledger.observe(cursor)
                if release_source:
                    self._ledger.forget_sources(frozenset({source}))
                return observed
            except BaseException:
                self._failed = True
                raise

    async def observe(self) -> dict[SourceKey, SourcePlaybackCursor]:
        async with self._operation_lock:
            self._require_healthy()
            try:
                return self._ledger.observe(self._master.cursor)
            except BaseException:
                self._failed = True
                raise

    async def forget_sources(self, sources: frozenset[SourceKey]) -> int:
        """Release source-local cursor history after its caller has consumed it."""
        async with self._operation_lock:
            self._require_healthy()
            return self._ledger.forget_sources(sources)

    async def stats(self) -> AudioMixerBusStats:
        """Return a coherent, credential-free snapshot without exposing internals."""
        async with self._operation_lock:
            async with self._condition:
                buffered_frames = 0 if self._closed else self._estimated_master_buffered_frames()
                return AudioMixerBusStats(
                    master_buffered_ms=buffered_frames * 1_000 / AUDIO_SAMPLE_RATE,
                    master_max_buffered_ms=(
                        self._master_max_buffered_frames * 1_000 / AUDIO_SAMPLE_RATE
                    ),
                    master_underrun_count=self._master_underrun_count,
                    mixer_deadline_miss_count=self._mixer_deadline_miss_count,
                    music_queue_ms=(
                        len(self._queues[AudioSourceKind.MUSIC]) * AUDIO_BLOCK_DURATION_MS
                    ),
                    voice_queue_ms=(
                        len(self._queues[AudioSourceKind.VOICE]) * AUDIO_BLOCK_DURATION_MS
                    ),
                    remix_count=self._remix_count,
                    last_remix_ms=self._last_remix_ms,
                    max_remix_ms=self._max_remix_ms,
                    replayed_music_ms=(
                        self._replayed_frames[AudioSourceKind.MUSIC] * 1_000 / AUDIO_SAMPLE_RATE
                    ),
                    replayed_voice_ms=(
                        self._replayed_frames[AudioSourceKind.VOICE] * 1_000 / AUDIO_SAMPLE_RATE
                    ),
                    limiter_active_blocks=self._limiter_active_blocks,
                    max_gain_reduction_db=self._max_gain_reduction_db,
                    hard_clip_samples=self._hard_clip_samples,
                    retained_source_count=self._ledger.source_count,
                    ledger_entry_count=self._ledger.entry_count,
                )

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            async with self._condition:
                self._condition.notify_all()
            if not self._writer.done():
                self._writer.cancel()
            await asyncio.gather(self._writer, return_exceptions=True)
            queued = await self._take_all_queued()
            self._fail_waiters(queued, AudioBusFailedError("Shared audio bus is closing"))
            await self._master.aclose()
            self._closed = True
            final_stats = await self.stats()
            logger.info("Closed shared audio master: metrics=%s", final_stats.as_metrics())

    async def _writer_loop(self) -> None:
        try:
            while True:
                async with self._condition:
                    while not self._closing and not any(self._queues.values()):
                        await self._condition.wait()
                    if self._closing:
                        return
                async with self._operation_lock:
                    music, voice = await self._select_pair()
                    if music is None and voice is None:
                        continue
                    items = tuple(item for item in (music, voice) if item is not None)
                    try:
                        cursors = await self._submit(
                            music.block if music is not None else None,
                            voice.block if voice is not None else None,
                        )
                    except asyncio.CancelledError:
                        self._fail_waiters(
                            items,
                            AudioBusFailedError("Shared audio writer was cancelled"),
                        )
                        raise
                    except Exception as exc:
                        self._failed = True
                        self._fail_waiters(items, exc)
                        queued = await self._take_all_queued()
                        self._fail_waiters(queued, exc)
                        return
                    for item in items:
                        if not item.accepted.done():
                            item.accepted.set_result(cursors)
        except asyncio.CancelledError:
            raise

    async def _select_pair(self) -> tuple[_QueuedBlock | None, _QueuedBlock | None]:
        async with self._condition:
            music_queue = self._queues[AudioSourceKind.MUSIC]
            voice_queue = self._queues[AudioSourceKind.VOICE]
            if not music_queue and not voice_queue:
                return None, None
            if bool(music_queue) != bool(voice_queue) and self._other_lane_active(
                AudioSourceKind.MUSIC if music_queue else AudioSourceKind.VOICE
            ):
                try:
                    async with asyncio.timeout(AUDIO_BLOCK_DURATION_MS / 1_000):
                        while not self._closing and not (music_queue and voice_queue):
                            await self._condition.wait()
                except TimeoutError:
                    pass
            music = music_queue.popleft() if music_queue else None
            voice = voice_queue.popleft() if voice_queue else None
            self._condition.notify_all()
            return music, voice

    def _other_lane_active(self, available: AudioSourceKind) -> bool:
        return self._other_participant_active(available)

    def _other_participant_active(self, source: AudioSourceKind) -> bool:
        participant = (
            VoiceParticipantKind.CONVERSATION
            if source is AudioSourceKind.MUSIC
            else VoiceParticipantKind.MUSIC
        )
        return participant in self._participants

    async def _submit(
        self,
        music: AudioBlock | None,
        voice: AudioBlock | None,
        *,
        transition_from: np.ndarray | None = None,
    ) -> dict[SourceKey, SourcePlaybackCursor]:
        await self._pace_master()
        started_at = time.monotonic()
        snapshot = DuckingSnapshot(
            conversation_active=(VoiceParticipantKind.CONVERSATION in self._participants),
            reasons=self._ducking_reasons,
        )
        mixed = self._mixer.mix(music, voice, snapshot)
        if mixed.limiter.gain_reduction_db > 0:
            self._limiter_active_blocks += 1
        self._max_gain_reduction_db = max(
            self._max_gain_reduction_db,
            mixed.limiter.gain_reduction_db,
        )
        self._hard_clip_samples += mixed.limiter.hard_clipped_samples
        if transition_from is not None:
            mixed = self._apply_transition(mixed, transition_from)
        entry_id = self._ledger.register_pending(mixed)
        master_cursor = await self._master.write(mixed.pcm_s16le)
        self._master_cursor_observed_at = time.monotonic()
        self._master_started = True
        self._master_max_buffered_frames = max(
            self._master_max_buffered_frames,
            master_cursor.buffered_frames,
        )
        if time.monotonic() - started_at > AUDIO_BLOCK_DURATION_MS / 1_000:
            self._mixer_deadline_miss_count += 1
        return self._ledger.mark_accepted(entry_id, master_cursor)

    async def _pace_master(self) -> None:
        estimated_buffered = self._estimated_master_buffered_frames()
        if self._master_started and estimated_buffered == 0:
            self._master_underrun_count += 1
        excess = estimated_buffered + AUDIO_BLOCK_FRAMES - self._target_buffer_frames
        if excess > 0:
            await asyncio.sleep(excess / AUDIO_SAMPLE_RATE)

    def _estimated_master_buffered_frames(self) -> int:
        cursor = self._master.cursor
        elapsed_frames = int(
            (time.monotonic() - self._master_cursor_observed_at) * AUDIO_SAMPLE_RATE
        )
        return max(0, cursor.buffered_frames - elapsed_frames)

    def _record_remix(self, plan: RemixPlan, elapsed_ms: float) -> None:
        self._remix_count += 1
        self._last_remix_ms = elapsed_ms
        self._max_remix_ms = max(self._max_remix_ms, elapsed_ms)
        for survivor in plan.survivors:
            if survivor.music is not None:
                self._replayed_frames[AudioSourceKind.MUSIC] += survivor.music.frame_count
            if survivor.voice is not None:
                self._replayed_frames[AudioSourceKind.VOICE] += survivor.voice.frame_count

    async def _submit_replays(self, plan: RemixPlan) -> None:
        if not plan.survivors:
            return
        replay_blocks = self._replay_blocks(plan.survivors)
        transition = (
            plan.last_rendered_sample
            if plan.last_rendered_sample is not None
            else np.zeros(AUDIO_CHANNELS, dtype=np.float32)
        )
        for music, voice in replay_blocks:
            await self._submit(music, voice, transition_from=transition)
            transition = None

    @classmethod
    def _replay_blocks(
        cls,
        survivors: tuple[ReplaySegment, ...],
    ) -> tuple[tuple[AudioBlock | None, AudioBlock | None], ...]:
        kinds = {
            source.kind
            for survivor in survivors
            for source in (survivor.music, survivor.voice)
            if source is not None
        }
        if len(kinds) == 1 and all(
            (survivor.music is None) != (survivor.voice is None) for survivor in survivors
        ):
            kind = next(iter(kinds))
            slices = tuple(
                survivor.music if kind is AudioSourceKind.MUSIC else survivor.voice
                for survivor in survivors
            )
            blocks = cls._pack_single_lane(tuple(source for source in slices if source is not None))
            return tuple(
                (block, None) if kind is AudioSourceKind.MUSIC else (None, block)
                for block in blocks
            )
        return tuple(
            (cls._block_from_slice(item.music), cls._block_from_slice(item.voice))
            for item in survivors
        )

    @staticmethod
    def _pack_single_lane(slices: tuple[SourceSlice, ...]) -> tuple[AudioBlock, ...]:
        blocks: list[AudioBlock] = []
        pending: list[np.ndarray] = []
        pending_frames = 0
        pending_key: SourceKey | None = None
        pending_start = 0

        def emit(*, final: bool) -> None:
            nonlocal pending, pending_frames, pending_start
            while pending_frames >= AUDIO_BLOCK_FRAMES or (final and pending_frames):
                valid_frames = min(pending_frames, AUDIO_BLOCK_FRAMES)
                joined = np.concatenate(pending, axis=0)
                samples = np.zeros((AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS), dtype=np.float32)
                samples[:valid_frames] = joined[:valid_frames]
                assert pending_key is not None
                blocks.append(
                    AudioBlock(
                        pending_key.source_id,
                        pending_key.kind,
                        pending_key.generation,
                        pending_start,
                        valid_frames,
                        samples,
                    )
                )
                remainder = joined[valid_frames:]
                pending = [remainder] if remainder.size else []
                pending_frames -= valid_frames
                pending_start += valid_frames

        for source in slices:
            if source.master_offset_frames:
                raise AudioLedgerError("Single-lane replay slice is not left aligned")
            key = SourceKey.from_slice(source)
            if pending_key is not None and (
                key != pending_key or source.source_start_frame != pending_start + pending_frames
            ):
                emit(final=True)
                pending_key = None
            if pending_key is None:
                pending_key = key
                pending_start = source.source_start_frame
            pending.append(source.samples)
            pending_frames += source.frame_count
            emit(final=False)
        emit(final=True)
        return tuple(blocks)

    @staticmethod
    def _apply_transition(mixed: MixedAudioBlock, previous: np.ndarray) -> MixedAudioBlock:
        transition_frames = min(AUDIO_SAMPLE_RATE * 5 // 1_000, AUDIO_BLOCK_FRAMES)
        samples = np.array(mixed.samples, dtype=np.float32, copy=True)
        start = np.asarray(previous, dtype=np.float32)
        if start.shape != (AUDIO_CHANNELS,):
            raise AudioLedgerError("Replay transition sample has an invalid shape")
        weights = np.linspace(0.0, 1.0, transition_frames + 1, dtype=np.float32)[1:]
        samples[:transition_frames] = (
            start[np.newaxis, :] * (1.0 - weights[:, np.newaxis])
            + samples[:transition_frames] * weights[:, np.newaxis]
        )
        samples.setflags(write=False)
        return MixedAudioBlock(
            samples,
            float32_stereo_to_s16le(samples),
            mixed.music,
            mixed.voice,
            mixed.limiter,
        )

    @staticmethod
    def _block_from_slice(source: SourceSlice | None) -> AudioBlock | None:
        if source is None:
            return None
        if source.master_offset_frames:
            raise AudioLedgerError("Replay source offset is not canonical block aligned")
        samples = np.zeros((AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS), dtype=np.float32)
        samples[: source.frame_count] = source.samples
        return AudioBlock(
            source.source_id,
            source.kind,
            source.generation,
            source.source_start_frame,
            source.frame_count,
            samples,
        )

    async def _discard_queued(self, discard: frozenset[SourceKey]) -> tuple[_QueuedBlock, ...]:
        removed: list[_QueuedBlock] = []
        async with self._condition:
            for kind, queue in self._queues.items():
                retained: deque[_QueuedBlock] = deque()
                while queue:
                    item = queue.popleft()
                    key = SourceKey(
                        item.block.source_id,
                        kind,
                        item.block.generation,
                    )
                    if key in discard:
                        removed.append(item)
                    else:
                        retained.append(item)
                self._queues[kind] = retained
            self._condition.notify_all()
        return tuple(removed)

    async def _take_all_queued(self) -> tuple[_QueuedBlock, ...]:
        async with self._condition:
            queued = tuple(item for queue in self._queues.values() for item in queue)
            for queue in self._queues.values():
                queue.clear()
            self._condition.notify_all()
            return queued

    @staticmethod
    def _fail_waiters(items: tuple[_QueuedBlock, ...], error: BaseException) -> None:
        for item in items:
            if not item.accepted.done():
                item.accepted.set_exception(error)

    def _require_healthy(self) -> None:
        if self._closing or self._closed:
            raise AudioBusFailedError("Shared audio bus is closed")
        if self._failed:
            raise AudioBusFailedError("Shared audio bus transport has failed")


def master_buffer_frames(max_buffer_ms: int) -> int:
    if max_buffer_ms <= 0:
        raise ValueError("Master buffer duration must be positive")
    return 48_000 * max_buffer_ms // 1_000
