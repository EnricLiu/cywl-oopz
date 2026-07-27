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


class BrowserError(CywlError):
    """Base class for expected browser lifecycle and operation failures."""


class WebPageUrlError(BrowserError):
    """Raised when a URL is outside the public HTTP(S) boundary."""


class BrowserUnavailableError(BrowserError):
    """Raised when the MCP server or browser process cannot serve a request."""


class BrowserTimeoutError(BrowserError):
    """Raised when an MCP operation exceeds its configured time budget."""


class BrowserNavigationError(BrowserError):
    """Raised when a public page cannot be opened or read."""


class BrowserStaleRefError(BrowserError):
    """Raised when an element reference no longer belongs to the current page."""


class BrowserActionError(BrowserError):
    """Raised when an allowed browser interaction fails."""


class BrowserContractError(BrowserError):
    """Raised when the installed MCP server does not match the pinned contract."""
