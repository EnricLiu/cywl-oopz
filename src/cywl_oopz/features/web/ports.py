"""Replaceable boundaries for web providers."""

from __future__ import annotations

from typing import Protocol

from .models import (
    BrowserActionResult,
    BrowserDocument,
    BrowserPageView,
    BrowserWaitRequest,
    WebSearchRequest,
    WebSearchResult,
)


class WebSearchGateway(Protocol):
    """Search the public internet without leaking provider-specific values."""

    async def search(self, request: WebSearchRequest) -> tuple[WebSearchResult, ...]:
        """Return ordered, normalized results."""

    async def aclose(self) -> None:
        """Release owned worker resources."""


class BrowserGateway(Protocol):
    """High-level public-web operations backed by a replaceable browser provider."""

    async def start(self) -> None:
        """Start the provider and validate its required operation contract."""

    async def restart(self) -> None:
        """Rebuild a failed provider transport once."""

    async def open(self, session: str, url: str) -> BrowserPageView:
        """Navigate and return the new page snapshot."""

    async def read(self, session: str, url: str | None) -> BrowserDocument:
        """Read a URL or the active page as bounded text."""

    async def snapshot(
        self,
        session: str,
        *,
        interactive: bool,
        compact: bool,
    ) -> BrowserPageView:
        """Return the current bounded accessibility snapshot."""

    async def wait(
        self,
        session: str,
        request: BrowserWaitRequest,
    ) -> BrowserPageView:
        """Wait for a supported condition and return fresh page state."""

    async def click(self, session: str, ref: str) -> BrowserPageView:
        """Click one current snapshot ref and return fresh page state."""

    async def fill(
        self,
        session: str,
        ref: str,
        text: str,
    ) -> BrowserActionResult:
        """Fill one current snapshot ref without submitting the page."""

    async def press(self, session: str, key: str) -> BrowserPageView:
        """Press one allowed key and return fresh page state."""

    async def close_session(self, session: str) -> None:
        """Close one project-owned browser session."""

    async def aclose(self) -> None:
        """Close the MCP transport owned by the application."""
