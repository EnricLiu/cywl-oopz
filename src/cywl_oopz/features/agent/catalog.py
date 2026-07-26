"""In-memory provider catalog loaded from PostgreSQL on explicit boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from uuid import UUID

from .models import AgentModelRef, LlmModel, LlmProvider, ModelCapability
from .ports import ProviderCatalogAdminRepository, ProviderCatalogRepository


@dataclass(frozen=True, slots=True)
class ProviderCatalog:
    """Validated immutable snapshot used by selection and model registries."""

    providers: Mapping[UUID, LlmProvider]
    models: Mapping[UUID, LlmModel]

    @classmethod
    def build(
        cls,
        providers: tuple[LlmProvider, ...],
        models: tuple[LlmModel, ...],
    ) -> ProviderCatalog:
        """Validate identities, aliases, ownership, and configured defaults."""
        provider_map = {provider.id: provider for provider in providers}
        model_map = {model.id: model for model in models}
        if len(provider_map) != len(providers) or len(model_map) != len(models):
            raise ValueError("Provider catalog contains duplicate IDs")
        if len({provider.alias for provider in providers}) != len(providers):
            raise ValueError("Provider catalog contains duplicate aliases")

        aliases: set[tuple[UUID, str]] = set()
        application_defaults = 0
        provider_defaults: set[UUID] = set()
        for model in models:
            if model.provider_id not in provider_map:
                raise ValueError("Model references an unknown provider")
            alias_key = (model.provider_id, model.alias)
            if alias_key in aliases:
                raise ValueError("Provider catalog contains duplicate model aliases")
            aliases.add(alias_key)
            if model.fallback_model_id is not None and model.fallback_model_id not in model_map:
                raise ValueError("Model fallback references an unknown model")
            if model.is_application_default:
                application_defaults += 1
            if model.is_provider_default:
                if model.provider_id in provider_defaults:
                    raise ValueError("Provider has more than one default model")
                provider_defaults.add(model.provider_id)
        if application_defaults > 1:
            raise ValueError("Provider catalog has more than one application default")

        return cls(
            providers=MappingProxyType(provider_map),
            models=MappingProxyType(model_map),
        )

    def application_default_model_id(self) -> UUID | None:
        """Return the configured global default without guessing from aliases."""
        return next(
            (model.id for model in self.models.values() if model.is_application_default),
            None,
        )

    def resolve(
        self,
        model_id: UUID,
        *,
        required_capabilities: frozenset[ModelCapability],
        require_user_selectable: bool,
    ) -> AgentModelRef | None:
        """Resolve an enabled, compatible model into a credential-free run reference."""
        model = self.models.get(model_id)
        if model is None or not model.enabled:
            return None
        provider = self.providers.get(model.provider_id)
        if provider is None or not provider.enabled:
            return None
        if require_user_selectable and not provider.user_selectable:
            return None
        if not required_capabilities.issubset(model.capabilities):
            return None
        return AgentModelRef(
            provider_id=provider.id,
            model_id=model.id,
            provider_alias=provider.alias,
            model_alias=model.alias,
            remote_model_name=model.remote_model_name,
            protocol=provider.protocol,
            capabilities=model.capabilities,
            fallback_model_id=model.fallback_model_id,
        )

    def find_selectable(
        self,
        provider_alias: str,
        model_alias: str | None = None,
        *,
        required_capabilities: frozenset[ModelCapability] = frozenset(),
    ) -> AgentModelRef | None:
        """Resolve an owner-defined alias, using the Provider default when omitted."""
        provider_name = provider_alias.strip().casefold()
        model_name = model_alias.strip().casefold() if model_alias is not None else None
        provider = next(
            (
                candidate
                for candidate in self.providers.values()
                if candidate.alias.casefold() == provider_name
            ),
            None,
        )
        if provider is None or not provider.user_selectable:
            return None
        candidates = [
            model
            for model in self.models.values()
            if model.provider_id == provider.id
            and (
                model.alias.casefold() == model_name
                if model_name is not None
                else model.is_provider_default
            )
        ]
        if len(candidates) != 1:
            return None
        return self.resolve(
            candidates[0].id,
            required_capabilities=required_capabilities,
            require_user_selectable=True,
        )

    def selectable_models(
        self,
        *,
        required_capabilities: frozenset[ModelCapability] = frozenset(),
    ) -> tuple[AgentModelRef, ...]:
        """List enabled user-selectable models in stable alias order."""
        resolved = (
            self.resolve(
                model.id,
                required_capabilities=required_capabilities,
                require_user_selectable=True,
            )
            for model in self.models.values()
        )
        return tuple(
            sorted(
                (model for model in resolved if model is not None),
                key=lambda item: (item.provider_alias, item.model_alias),
            )
        )


class ReloadableProviderCatalog:
    """Own one atomically replaced catalog snapshot."""

    def __init__(self, repository: ProviderCatalogRepository) -> None:
        self._repository = repository
        self._catalog = ProviderCatalog.build((), ())
        self._reload_lock = asyncio.Lock()

    @property
    def snapshot(self) -> ProviderCatalog:
        """Return the current immutable snapshot without holding a lock."""
        return self._catalog

    async def reload(self) -> ProviderCatalog:
        """Load and validate a replacement before publishing it."""
        async with self._reload_lock:
            providers, models = await asyncio.gather(
                self._repository.load_providers(),
                self._repository.load_models(),
            )
            replacement = ProviderCatalog.build(providers, models)
            self._catalog = replacement
            return replacement


class ProviderCatalogAdminService:
    """Register owner-supplied configuration and publish a fresh snapshot."""

    def __init__(
        self,
        repository: ProviderCatalogAdminRepository,
        catalog: ReloadableProviderCatalog,
    ) -> None:
        self._repository = repository
        self._catalog = catalog

    async def register(
        self,
        provider: LlmProvider,
        models: tuple[LlmModel, ...],
    ) -> ProviderCatalog:
        """Validate and atomically upsert one provider bundle, including API key."""
        if any(model.provider_id != provider.id for model in models):
            raise ValueError("All registered models must belong to the provider")
        ProviderCatalog.build((provider,), models)
        await self._repository.upsert_provider_bundle(provider, models)
        return await self._catalog.reload()
