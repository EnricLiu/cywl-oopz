"""Framework-neutral ports for privileged administration use cases."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from .models import (
    AgentResponseDiagnostic,
    AreaChannelCatalog,
    AreaInitializationResult,
    ChannelInitializationResult,
    ChannelKey,
    OopzMessageAddress,
    OutboundMessageReceipt,
    OutboundMessageState,
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


class OutboundMessageRepository(Protocol):
    """Best-effort persistence for Bot-owned OOPZ messages."""

    async def create(self, receipt: OutboundMessageReceipt) -> bool:
        """Insert one receipt and report whether it was new."""

    async def bind_agent_run(self, message_id: str, run_id: UUID) -> bool:
        """Link a newly-created Agent run to its already-visible reply."""

    async def promote_agent_response(
        self,
        message_id: str,
        run_id: UUID | None,
        diagnostic_snapshot: dict[str, object],
    ) -> bool:
        """Turn a generic tracked reply into a terminal Agent response."""

    async def update_state(
        self,
        message_id: str,
        state: OutboundMessageState,
        *,
        diagnostic_snapshot: dict[str, object] | None = None,
    ) -> bool:
        """Update lifecycle state and an optional display-safe diagnostic snapshot."""


class AgentDiagnosticRepository(Protocol):
    """Read a bounded diagnostic aggregate by exact tracked message address."""

    async def get_by_outbound_message(
        self,
        message_id: str,
        address: OopzMessageAddress,
    ) -> AgentResponseDiagnostic | None:
        """Return the matching Bot Agent response or none."""


class AgentDiagnosticRenderer(Protocol):
    """Render one immutable diagnostic into OOPZ-safe pages."""

    def render(
        self,
        diagnostic: AgentResponseDiagnostic,
        *,
        verbose: bool,
    ) -> tuple[str, ...]:
        """Return bounded pages without performing I/O."""
