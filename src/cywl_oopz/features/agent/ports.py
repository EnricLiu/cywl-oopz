"""Project-owned ports for Agent engines and persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.chat.progress import ProgressSink

from .models import (
    AgentMessage,
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

    async def run(
        self,
        request: AgentRunRequest,
        progress: ProgressSink | None = None,
    ) -> AgentRunResult:
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

    async def set_user_model(self, person_id: str, model_id: UUID) -> None:
        """Set the user's default for subsequently resolved threads."""


class AgentThreadRepository(Protocol):
    """Persistence boundary for Agent thread metadata."""

    async def get(self, key: ConversationKey) -> AgentThread | None:
        """Load one thread by its privacy scope."""

    async def add(self, thread: AgentThread) -> None:
        """Create a new thread."""

    async def set_selected_model(self, key: ConversationKey, model_id: UUID) -> None:
        """Pin an existing thread to one model."""

    async def refresh_expiry(self, thread_id: UUID, expires_at: datetime) -> None:
        """Extend one active thread TTL."""

    async def save_summary(
        self,
        thread_id: UUID,
        summary: str,
        through_sequence: int,
        *,
        expected_version: int,
    ) -> bool:
        """Save a derived summary with optimistic concurrency."""

    async def delete(self, key: ConversationKey) -> None:
        """Delete a thread and its cascading runtime records."""


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


class AgentMessageRepository(Protocol):
    """Persistence boundary for ordered provider-neutral thread messages."""

    async def load(
        self,
        thread_id: UUID,
        *,
        limit: int,
        after_sequence: int = 0,
    ) -> tuple[AgentMessage, ...]:
        """Load the newest messages in chronological order."""

    async def load_after(
        self,
        thread_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> tuple[AgentMessage, ...]:
        """Load the oldest messages after a sequence in chronological order."""

    async def append(
        self,
        thread_id: UUID,
        run_id: UUID,
        messages: tuple[AgentMessage, ...],
    ) -> None:
        """Append messages with thread-local monotonic sequence numbers."""

    async def count(self, thread_id: UUID) -> int:
        """Count messages without loading their contents."""
