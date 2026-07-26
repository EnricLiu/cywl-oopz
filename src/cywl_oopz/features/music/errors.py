"""Expected music failures that may be reduced to safe Agent tool errors."""

from __future__ import annotations

from cywl_oopz.core.errors import CywlError


class MusicError(CywlError):
    """Base class for expected catalog, queue, and playback failures."""


class MusicCatalogError(MusicError):
    """Raised when the configured catalog cannot return a valid result."""


class MusicQueryError(MusicError):
    """Raised when a music query is empty or exceeds project bounds."""


class MusicNotFoundError(MusicError):
    """Raised when a search or stream resolution has no playable result."""


class MusicVoiceChannelRequiredError(MusicError):
    """Raised when the caller is not currently in an OOPZ voice channel."""


class MusicQueueFullError(MusicError):
    """Raised when one voice channel reaches its bounded queue length."""


class MusicPlaybackError(MusicError):
    """Raised when the OOPZ voice backend cannot perform a requested action."""
