"""Replaceable boundaries for web providers."""

from __future__ import annotations

from typing import Protocol

from .models import WebSearchRequest, WebSearchResult


class WebSearchGateway(Protocol):
    """Search the public internet without leaking provider-specific values."""

    async def search(self, request: WebSearchRequest) -> tuple[WebSearchResult, ...]:
        """Return ordered, normalized results."""

    async def aclose(self) -> None:
        """Release owned worker resources."""
