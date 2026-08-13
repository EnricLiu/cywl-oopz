"""Framework-neutral ports for privileged administration use cases."""

from __future__ import annotations

from typing import Protocol

from .models import (
    AreaChannelCatalog,
    AreaInitializationResult,
    ChannelInitializationResult,
    ChannelKey,
)


class AreaChannelCatalogPort(Protocol):
    """Discover channels visible to the Bot without leaking SDK models."""

    async def discover(self, area_id: str) -> AreaChannelCatalog:
        """Return a typed, deduplicated area channel catalog."""


class ChannelInitializationRepository(Protocol):
    """Persist missing channel settings while preserving all existing rows."""

    async def initialize_text_channel(
        self,
        channel: ChannelKey,
    ) -> ChannelInitializationResult:
        """Insert one text settings row if it does not exist."""

    async def initialize_area(
        self,
        catalog: AreaChannelCatalog,
    ) -> AreaInitializationResult:
        """Insert missing text/voice settings together in one transaction."""
