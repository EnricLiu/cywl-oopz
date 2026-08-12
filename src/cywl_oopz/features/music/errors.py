"""Expected music failures that may be reduced to safe Agent tool errors."""

from __future__ import annotations

from cywl_oopz.core.errors import CywlError


class MusicError(CywlError):
    """Base class for expected catalog, queue, and playback failures."""


class MusicCatalogError(MusicError):
    """Raised when the configured catalog cannot return a valid result."""


class MusicSourceDisabledError(MusicCatalogError):
    """Raised when a valid request targets a provider not enabled by this deployment."""


class MusicSourceUnavailableError(MusicCatalogError):
    """Raised when an enabled provider cannot currently serve requests."""


class MusicExtractionTimeoutError(MusicSourceUnavailableError):
    """Raised when a bounded yt-dlp operation does not settle in time."""


class MusicExtractorProcessError(MusicSourceUnavailableError):
    """Raised when the isolated extractor worker exits without a valid response."""


class MusicExtractorProtocolError(MusicSourceUnavailableError):
    """Raised when the isolated worker violates the project JSON protocol."""


class MusicReferenceError(MusicError):
    """Raised when a music URL or stable provider identifier is invalid."""


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


class MusicBackendClosedError(MusicPlaybackError):
    """The shared voice transport closed and the current track may be retried once."""


class MusicDecoderError(MusicPlaybackError):
    """FFmpeg could not produce a valid bounded canonical PCM stream."""


class MusicDecoderUnavailableError(MusicDecoderError):
    """The configured FFmpeg executable is absent or incompatible."""


class MusicVoiceBusyError(MusicPlaybackError):
    """Raised when music cannot reserve the backend without preempting its owner."""


class MusicAreaRequiredError(MusicError):
    """Raised when a shared playlist operation has no OOPZ area scope."""


class MusicPlaylistNameError(MusicError):
    """Raised when a playlist name is empty or too long."""


class MusicPlaylistConflictError(MusicError):
    """Raised when an area already has a playlist with the requested name."""


class MusicPlaylistNotFoundError(MusicError):
    """Raised when a playlist is absent from the caller's area."""


class MusicPlaylistFullError(MusicError):
    """Raised when a playlist reaches the configured track capacity."""


class MusicPlaylistEmptyError(MusicError):
    """Raised when an empty playlist cannot rebuild a playback queue."""


class NeteasePlaylistReferenceError(MusicError):
    """Raised when a Netease playlist ID or URL cannot be parsed safely."""


class NeteasePlaylistNotFoundError(MusicError):
    """Raised when Netease does not expose the requested playlist."""


class NeteasePlaylistIncompleteError(MusicError):
    """Raised before an incomplete source playlist is imported without consent."""


class NeteasePlaylistTooLargeError(MusicError):
    """Raised when a complete source playlist exceeds area playlist capacity."""
