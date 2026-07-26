from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from cywl_oopz.core.errors import ProviderSelectionError
from cywl_oopz.features.agent.catalog import ProviderCatalog, ReloadableProviderCatalog
from cywl_oopz.features.agent.models import (
    AgentRunState,
    AgentRunStatus,
    AgentStopReason,
    LlmModel,
    LlmProvider,
    ModelCapability,
    ModelSelectionCandidates,
    ModelSelectionSource,
    ProviderProtocol,
)
from cywl_oopz.features.agent.selection import ProviderSelectionService
from cywl_oopz.features.chat.models import ConversationKey

PROVIDER_ID = UUID("00000000-0000-0000-0000-000000000001")
MODEL_ID = UUID("00000000-0000-0000-0000-000000000002")
FALLBACK_MODEL_ID = UUID("00000000-0000-0000-0000-000000000003")


def provider(*, selectable: bool = True, enabled: bool = True) -> LlmProvider:
    return LlmProvider(
        id=PROVIDER_ID,
        alias="primary",
        display_name="Primary",
        protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
        base_url="https://llm.example/v1/",
        api_key="db-key",
        user_selectable=selectable,
        enabled=enabled,
    )


def model(
    model_id: UUID = MODEL_ID,
    *,
    enabled: bool = True,
    application_default: bool = False,
    capabilities: frozenset[ModelCapability] = frozenset({ModelCapability.TOOL_CALLING}),
) -> LlmModel:
    return LlmModel(
        id=model_id,
        provider_id=PROVIDER_ID,
        alias=f"model-{model_id.int}",
        remote_model_name=f"remote-{model_id.int}",
        display_name=f"Model {model_id.int}",
        enabled=enabled,
        is_provider_default=model_id == MODEL_ID,
        is_application_default=application_default,
        capabilities=capabilities,
    )


class FakeCatalogRepository:
    def __init__(
        self,
        providers: tuple[LlmProvider, ...],
        models: tuple[LlmModel, ...],
    ) -> None:
        self.providers = providers
        self.models = models

    async def load_providers(self) -> tuple[LlmProvider, ...]:
        return self.providers

    async def load_models(self) -> tuple[LlmModel, ...]:
        return self.models


class FakeSelectionRepository:
    def __init__(self, candidates: ModelSelectionCandidates) -> None:
        self.candidates = candidates

    async def load_candidates(self, _: ConversationKey) -> ModelSelectionCandidates:
        return self.candidates


@pytest.mark.asyncio
async def test_selection_falls_back_from_disabled_thread_to_user_default() -> None:
    catalog = ReloadableProviderCatalog(
        FakeCatalogRepository(
            (provider(),),
            (
                model(MODEL_ID, enabled=False),
                model(FALLBACK_MODEL_ID),
            ),
        )
    )
    await catalog.reload()
    selections = FakeSelectionRepository(
        ModelSelectionCandidates(
            thread_model_id=MODEL_ID,
            user_model_id=FALLBACK_MODEL_ID,
        )
    )

    selected = await ProviderSelectionService(catalog, selections).resolve(
        ConversationKey("private", "", "", "person"),
        required_capabilities=frozenset({ModelCapability.TOOL_CALLING}),
    )

    assert selected.model.model_id == FALLBACK_MODEL_ID
    assert selected.source is ModelSelectionSource.USER
    assert selected.skipped_sources == (ModelSelectionSource.THREAD,)


@pytest.mark.asyncio
async def test_user_selection_requires_selectable_provider_and_capabilities() -> None:
    catalog = ReloadableProviderCatalog(
        FakeCatalogRepository(
            (provider(selectable=False),),
            (model(MODEL_ID, application_default=True),),
        )
    )
    await catalog.reload()
    selections = FakeSelectionRepository(ModelSelectionCandidates(user_model_id=MODEL_ID))

    selected = await ProviderSelectionService(catalog, selections).resolve(
        ConversationKey("private", "", "", "person"),
        required_capabilities=frozenset({ModelCapability.TOOL_CALLING}),
    )

    assert selected.source is ModelSelectionSource.APPLICATION
    assert selected.skipped_sources == (ModelSelectionSource.USER,)


@pytest.mark.asyncio
async def test_selection_fails_when_no_model_has_required_capability() -> None:
    catalog = ReloadableProviderCatalog(
        FakeCatalogRepository(
            (provider(),),
            (model(MODEL_ID, application_default=True, capabilities=frozenset()),),
        )
    )
    await catalog.reload()

    with pytest.raises(ProviderSelectionError):
        await ProviderSelectionService(
            catalog,
            FakeSelectionRepository(ModelSelectionCandidates()),
        ).resolve(
            ConversationKey("private", "", "", "person"),
            required_capabilities=frozenset({ModelCapability.TOOL_CALLING}),
        )


def test_catalog_rejects_duplicate_application_defaults() -> None:
    with pytest.raises(ValueError, match="application default"):
        ProviderCatalog.build(
            (provider(),),
            (
                model(MODEL_ID, application_default=True),
                model(FALLBACK_MODEL_ID, application_default=True),
            ),
        )


def test_run_state_maps_terminal_reasons_and_rejects_reentry() -> None:
    now = datetime.now(UTC)
    running = AgentRunState(uuid4()).start(now)
    finished = running.finish(AgentStopReason.COMPLETED, now)

    assert finished.status is AgentRunStatus.SUCCEEDED
    assert finished.stop_reason is AgentStopReason.COMPLETED
    with pytest.raises(ValueError, match="Cannot finish"):
        finished.finish(AgentStopReason.CANCELLED, now)
