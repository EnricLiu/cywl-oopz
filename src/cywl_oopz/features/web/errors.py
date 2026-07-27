"""Expected web feature failures exposed through stable Agent tool errors."""

from __future__ import annotations

from cywl_oopz.core.errors import CywlError


class WebSearchError(CywlError):
    """Base class for bounded internet-search failures."""


class WebSearchQueryError(WebSearchError):
    """Raised when a search query does not satisfy project bounds."""


class WebSearchTimeoutError(WebSearchError):
    """Raised when DuckDuckGo does not respond inside its time budget."""


class WebSearchRateLimitError(WebSearchError):
    """Raised when DuckDuckGo temporarily rejects search traffic."""


class WebSearchUnavailableError(WebSearchError):
    """Raised when a search provider returns no usable response."""
