"""Provider/model selection with explicit, testable precedence."""

from __future__ import annotations

import logging

from cywl_oopz.core.errors import ProviderSelectionError
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.chat.models import ConversationKey

from .catalog import ReloadableProviderCatalog
from .models import ModelCapability, ModelSelection, ModelSelectionSource
from .ports import ModelSelectionRepository

logger = logging.getLogger(__name__)


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
        await self._catalog.refresh_if_stale()
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
                logger.debug(
                    "Agent model resolved: conversation=%s source=%s model=%s/%s skipped=%s",
                    opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id),
                    source.value,
                    model.provider_alias,
                    model.model_alias,
                    len(skipped),
                )
                return ModelSelection(model, source, tuple(skipped))
            skipped.append(source)

        logger.warning(
            "No Agent model satisfies selection: conversation=%s required_capabilities=%s",
            opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id),
            ",".join(sorted(capability.value for capability in required_capabilities)) or "none",
        )
        raise ProviderSelectionError("No enabled LLM model satisfies this Agent run")
