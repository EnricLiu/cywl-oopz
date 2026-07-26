"""Provider/model selection with explicit, testable precedence."""

from __future__ import annotations

from cywl_oopz.core.errors import ProviderSelectionError
from cywl_oopz.features.chat.models import ConversationKey

from .catalog import ReloadableProviderCatalog
from .models import ModelCapability, ModelSelection, ModelSelectionSource
from .ports import ModelSelectionRepository


class ProviderSelectionService:
    """Resolve one run-pinned model from durable user and channel preferences."""

    def __init__(
        self,
        catalog: ReloadableProviderCatalog,
        selections: ModelSelectionRepository,
    ) -> None:
        self._catalog = catalog
        self._selections = selections

    async def resolve(
        self,
        key: ConversationKey,
        *,
        required_capabilities: frozenset[ModelCapability] = frozenset(),
    ) -> ModelSelection:
        """Apply thread → user → channel → application precedence with fallback."""
        candidates = await self._selections.load_candidates(key)
        catalog = self._catalog.snapshot
        application_default = (
            candidates.application_model_id or catalog.application_default_model_id()
        )
        ordered = list(candidates.in_precedence_order())
        ordered[-1] = (ModelSelectionSource.APPLICATION, application_default)
        skipped: list[ModelSelectionSource] = []

        for source, model_id in ordered:
            if model_id is None:
                continue
            model = catalog.resolve(
                model_id,
                required_capabilities=required_capabilities,
                require_user_selectable=source
                in {ModelSelectionSource.THREAD, ModelSelectionSource.USER},
            )
            if model is not None:
                return ModelSelection(model, source, tuple(skipped))
            skipped.append(source)

        raise ProviderSelectionError("No enabled LLM model satisfies this Agent run")
