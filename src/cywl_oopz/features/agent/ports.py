"""Project-owned ports for Agent engines and persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from cywl_oopz.features.chat.models import ConversationKey

from .models import (
    AgentRun,
    AgentRunRequest,
    AgentRunResult,
    AgentRunState,
    AgentThread,
    LlmModel,
    LlmProvider,
    ModelSelectionCandidates,
)


class AgentEngine(Protocol):
    """Framework-neutral boundary for a bounded Agent implementation."""

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        """Run one already-pinned request."""

    async def aclose(self) -> None:
        """Close all engine-owned asynchronous resources."""


class ProviderCatalogRepository(Protocol):
    """Persistence boundary for provider and model configuration."""

    async def load_providers(self) -> tuple[LlmProvider, ...]:
        """Load the complete provider directory."""

    async def load_models(self) -> tuple[LlmModel, ...]:
        """Load the complete model directory."""


class ProviderCatalogAdminRepository(Protocol):
    """Owner-only mutation boundary for provider/model configuration."""

    async def upsert_provider_bundle(
        self,
        provider: LlmProvider,
        models: tuple[LlmModel, ...],
    ) -> None:
        """Atomically register one provider and its supplied models."""


class ModelSelectionRepository(Protocol):
    """Persistence boundary for model-selection precedence."""

    async def load_candidates(self, key: ConversationKey) -> ModelSelectionCandidates:
        """Load thread, user, channel, and application model IDs."""


class AgentThreadRepository(Protocol):
    """Persistence boundary for Agent thread metadata."""

    async def get(self, key: ConversationKey) -> AgentThread | None:
        """Load one thread by its privacy scope."""

    async def add(self, thread: AgentThread) -> None:
        """Create a new thread."""


class AgentRunRepository(Protocol):
    """Persistence boundary for short run lifecycle transactions."""

    async def add(self, run: AgentRun) -> None:
        """Persist a run that has already entered the running state."""

    async def finish(
        self,
        state: AgentRunState,
        *,
        usage: dict[str, object],
        error_code: str = "",
    ) -> None:
        """Persist one terminal state."""

    async def abandon_stale(self, before: datetime, now: datetime) -> int:
        """Mark stale running records abandoned and return the changed count."""
