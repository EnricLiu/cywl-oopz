"""Typed failures raised by the shared audio core."""


class AudioCoreError(Exception):
    """Base class for project-owned audio processing failures."""


class AudioFormatError(AudioCoreError, ValueError):
    """PCM bytes or arrays do not match their declared format."""


class AudioQueueClosedError(AudioCoreError):
    """A source queue no longer accepts or returns audio."""


class AudioBackpressureError(AudioCoreError):
    """A producer exceeded the bounded source queue wait budget."""


class AudioLedgerError(AudioCoreError):
    """Master/source cursor state violates a playout ledger invariant."""


class AudioLedgerCapacityError(AudioLedgerError):
    """Unrendered master audio exceeded the configured ledger window."""
