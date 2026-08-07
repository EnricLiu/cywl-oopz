"""Map master PCM epochs back to source-local accepted and rendered frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

import numpy as np

from .errors import AudioLedgerCapacityError, AudioLedgerError
from .mixer import MixedAudioBlock
from .models import (
    AUDIO_BLOCK_FRAMES,
    AUDIO_CHANNELS,
    AudioSourceKind,
    MasterPlaybackCursor,
    SourcePlaybackCursor,
    SourceSlice,
)


@dataclass(frozen=True, slots=True)
class SourceKey:
    """Stable identity of exactly one source generation."""

    source_id: UUID
    kind: AudioSourceKind
    generation: int

    @classmethod
    def from_slice(cls, source: SourceSlice) -> SourceKey:
        return cls(source.source_id, source.kind, source.generation)


class LedgerEntryState(StrEnum):
    """Whether the SDK acknowledged one registered master block."""

    PENDING = "pending"
    ACCEPTED = "accepted"


@dataclass(slots=True)
class MasterLedgerEntry:
    """One master block and its pre-gain source contributions."""

    entry_id: int
    epoch: int
    master_start_frame: int
    frame_count: int
    output_samples: np.ndarray
    music: SourceSlice | None
    voice: SourceSlice | None
    state: LedgerEntryState = LedgerEntryState.PENDING

    @property
    def master_end_frame(self) -> int:
        return self.master_start_frame + self.frame_count

    @property
    def sources(self) -> tuple[SourceSlice, ...]:
        return tuple(source for source in (self.music, self.voice) if source is not None)


@dataclass(slots=True)
class _SourceProgress:
    accepted_frames: int = 0
    rendered_frames: int = 0


@dataclass(frozen=True, slots=True)
class ReplaySegment:
    """Unrendered source slices retained across an SDK master flush."""

    frame_count: int
    music: SourceSlice | None
    voice: SourceSlice | None

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("Replay segment must contain master frames")
        sources = tuple(source for source in (self.music, self.voice) if source is not None)
        if not sources:
            raise ValueError("Replay segment must retain at least one source")
        if any(source.master_end_offset_frames > self.frame_count for source in sources):
            raise ValueError("Replay source exceeds its master segment")


@dataclass(frozen=True, slots=True)
class RemixPlan:
    """Source-aware survivors and old cursors produced by one master flush."""

    old_epoch: int
    next_epoch: int
    source_cursors: dict[SourceKey, SourcePlaybackCursor]
    survivors: tuple[ReplaySegment, ...]
    last_rendered_sample: np.ndarray | None


class MasterPlayoutLedger:
    """Bound unrendered master data and reconcile it with SDK cursor ACKs."""

    def __init__(self, *, max_buffer_frames: int) -> None:
        if max_buffer_frames < AUDIO_BLOCK_FRAMES:
            raise ValueError("Master ledger must hold at least one audio block")
        self._max_buffer_frames = max_buffer_frames
        self._epoch = 0
        self._next_master_frame = 0
        self._master_accepted_frames = 0
        self._master_rendered_frames = 0
        self._next_entry_id = 0
        self._pending_entry_id: int | None = None
        self._entries: deque[MasterLedgerEntry] = deque()
        self._sources: dict[SourceKey, _SourceProgress] = {}
        self._last_rendered_sample: np.ndarray | None = None

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def pending_entry_id(self) -> int | None:
        return self._pending_entry_id

    @property
    def unrendered_frames(self) -> int:
        return self._next_master_frame - self._master_rendered_frames

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def register_pending(self, mixed: MixedAudioBlock) -> int:
        """Register source data before the single writer calls SDK ``write``."""
        if self._pending_entry_id is not None:
            raise AudioLedgerError("Only one master write may be pending")
        if mixed.samples.shape != (AUDIO_BLOCK_FRAMES, AUDIO_CHANNELS):
            raise AudioLedgerError("Ledger accepts exact canonical master blocks")
        if self.unrendered_frames + AUDIO_BLOCK_FRAMES > self._max_buffer_frames:
            raise AudioLedgerCapacityError("Master playout ledger window is full")
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        entry = MasterLedgerEntry(
            entry_id,
            self._epoch,
            self._next_master_frame,
            AUDIO_BLOCK_FRAMES,
            mixed.samples,
            mixed.music,
            mixed.voice,
        )
        for source in entry.sources:
            self._note_source_accepted(source)
        self._entries.append(entry)
        self._next_master_frame = entry.master_end_frame
        self._pending_entry_id = entry_id
        return entry_id

    def mark_accepted(
        self,
        entry_id: int,
        cursor: MasterPlaybackCursor,
    ) -> dict[SourceKey, SourcePlaybackCursor]:
        """Commit one pending block and advance rendered source cursors."""
        if self._pending_entry_id != entry_id:
            raise AudioLedgerError("Master ACK does not own the pending ledger entry")
        entry = self._entry(entry_id)
        if cursor.epoch != self._epoch or cursor.accepted_frames < entry.master_end_frame:
            raise AudioLedgerError("Master ACK does not include the pending block")
        entry.state = LedgerEntryState.ACCEPTED
        self._pending_entry_id = None
        self._reconcile(cursor)
        return self.source_cursors()

    def observe(
        self,
        cursor: MasterPlaybackCursor,
    ) -> dict[SourceKey, SourcePlaybackCursor]:
        """Apply a later stats/drain cursor without registering another block."""
        self._reconcile(cursor)
        return self.source_cursors()

    def source_cursor(self, key: SourceKey) -> SourcePlaybackCursor:
        progress = self._sources.get(key)
        if progress is None:
            raise KeyError(key)
        return SourcePlaybackCursor(
            key.generation,
            progress.accepted_frames,
            progress.rendered_frames,
        )

    def source_cursors(self) -> dict[SourceKey, SourcePlaybackCursor]:
        return {key: self.source_cursor(key) for key in self._sources}

    def remix(
        self,
        cursor: MasterPlaybackCursor,
        *,
        discard: frozenset[SourceKey] = frozenset(),
    ) -> RemixPlan:
        """Close one master epoch and retain only valid unrendered source slices."""
        old_epoch = self._epoch
        self._reconcile(cursor, allow_pending_acceptance=True)
        old_cursors = self.source_cursors()
        survivors: list[ReplaySegment] = []
        for entry in self._entries:
            prefix = max(0, self._master_rendered_frames - entry.master_start_frame)
            if prefix >= entry.frame_count:
                continue
            music = self._surviving_slice(entry.music, prefix, discard)
            voice = self._surviving_slice(entry.voice, prefix, discard)
            retained = tuple(source for source in (music, voice) if source is not None)
            if not retained:
                continue
            frame_count = max(source.master_end_offset_frames for source in retained)
            survivors.append(ReplaySegment(frame_count, music, voice))

        last_sample = (
            None
            if self._last_rendered_sample is None
            else np.array(self._last_rendered_sample, dtype=np.float32, copy=True)
        )
        if last_sample is not None:
            last_sample.setflags(write=False)
        self._epoch += 1
        self._next_master_frame = 0
        self._master_accepted_frames = 0
        self._master_rendered_frames = 0
        self._pending_entry_id = None
        self._entries.clear()
        self._last_rendered_sample = last_sample
        return RemixPlan(
            old_epoch,
            self._epoch,
            old_cursors,
            tuple(survivors),
            last_sample,
        )

    def _note_source_accepted(self, source: SourceSlice) -> None:
        key = SourceKey.from_slice(source)
        progress = self._sources.setdefault(key, _SourceProgress())
        if source.source_start_frame > progress.accepted_frames:
            raise AudioLedgerError("Source contribution introduced an unaccounted frame gap")
        progress.accepted_frames = max(progress.accepted_frames, source.source_end_frame)

    def _reconcile(
        self,
        cursor: MasterPlaybackCursor,
        *,
        allow_pending_acceptance: bool = False,
    ) -> None:
        if cursor.epoch != self._epoch:
            raise AudioLedgerError("Master cursor belongs to another ledger epoch")
        if cursor.accepted_frames < self._master_accepted_frames:
            raise AudioLedgerError("Master accepted cursor moved backwards")
        if cursor.rendered_frames < self._master_rendered_frames:
            raise AudioLedgerError("Master rendered cursor moved backwards")
        if cursor.accepted_frames > self._next_master_frame:
            raise AudioLedgerError("Master accepted cursor exceeds registered audio")
        if cursor.rendered_frames > cursor.accepted_frames:
            raise AudioLedgerError("Master rendered cursor exceeds accepted audio")

        pending = self._pending_entry_id
        for entry in self._entries:
            if entry.master_end_frame <= cursor.accepted_frames:
                entry.state = LedgerEntryState.ACCEPTED
                if entry.entry_id == pending and allow_pending_acceptance:
                    self._pending_entry_id = None
        if pending is not None and not allow_pending_acceptance:
            pending_entry = self._entry(pending)
            if pending_entry.master_end_frame <= cursor.accepted_frames:
                raise AudioLedgerError("Pending master write must be committed by its owner ACK")

        if cursor.rendered_frames > self._master_rendered_frames:
            self._advance_sources(cursor.rendered_frames)
            self._capture_last_rendered_sample(cursor.rendered_frames)
        self._master_accepted_frames = cursor.accepted_frames
        self._master_rendered_frames = cursor.rendered_frames
        while self._entries and self._entries[0].master_end_frame <= cursor.rendered_frames:
            self._entries.popleft()

    def _advance_sources(self, rendered_frames: int) -> None:
        for entry in self._entries:
            if entry.master_start_frame >= rendered_frames:
                break
            for source in entry.sources:
                source_master_start = entry.master_start_frame + source.master_offset_frames
                rendered_in_source = min(
                    source.frame_count,
                    max(0, rendered_frames - source_master_start),
                )
                if not rendered_in_source:
                    continue
                key = SourceKey.from_slice(source)
                progress = self._sources[key]
                progress.rendered_frames = max(
                    progress.rendered_frames,
                    source.source_start_frame + rendered_in_source,
                )
                if progress.rendered_frames > progress.accepted_frames:
                    raise AudioLedgerError("Source rendered cursor exceeded accepted audio")

    def _capture_last_rendered_sample(self, rendered_frames: int) -> None:
        sample_position = rendered_frames - 1
        for entry in self._entries:
            if entry.master_start_frame <= sample_position < entry.master_end_frame:
                offset = sample_position - entry.master_start_frame
                self._last_rendered_sample = np.array(
                    entry.output_samples[offset],
                    dtype=np.float32,
                    copy=True,
                )
                return

    @staticmethod
    def _surviving_slice(
        source: SourceSlice | None,
        prefix: int,
        discard: frozenset[SourceKey],
    ) -> SourceSlice | None:
        if source is None or SourceKey.from_slice(source) in discard:
            return None
        return source.trim_master_prefix(prefix)

    def _entry(self, entry_id: int) -> MasterLedgerEntry:
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        raise AudioLedgerError("Unknown master ledger entry")
