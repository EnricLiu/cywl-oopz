"""Provider registry and source-neutral catalog routing."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from .errors import MusicSourceDisabledError
from .models import (
    MusicPageLocator,
    MusicProviderHealth,
    MusicSourceKind,
    MusicTrack,
    MusicTrackReference,
    PlayableTrack,
)
from .ports import MusicProvider


class MusicProviderRegistry:
    """Immutable source-to-provider mapping owned by the composition root."""

    def __init__(self, providers: Iterable[MusicProvider]) -> None:
        registered: dict[MusicSourceKind, MusicProvider] = {}
        for provider in providers:
            source = MusicSourceKind(provider.source)
            if source in registered:
                raise ValueError(f"Duplicate music provider: {source.value}")
            registered[source] = provider
        if not registered:
            raise ValueError("At least one music provider is required")
        self._providers = registered

    @property
    def sources(self) -> tuple[MusicSourceKind, ...]:
        """Return enabled sources in deterministic declaration order."""
        return tuple(self._providers)

    def get(self, source: MusicSourceKind) -> MusicProvider:
        """Return one enabled provider or a stable expected failure."""
        normalized = MusicSourceKind(source)
        try:
            return self._providers[normalized]
        except KeyError as exc:
            raise MusicSourceDisabledError(
                f"Music source is not enabled: {normalized.value}"
            ) from exc

    async def health(self) -> tuple[MusicProviderHealth, ...]:
        """Read all provider health values without serial network waits."""
        pending = (provider.health() for provider in self._providers.values())
        return tuple(await asyncio.gather(*pending))

    async def aclose(self) -> None:
        """Close each registered provider exactly once."""
        await asyncio.gather(
            *(provider.aclose() for provider in self._providers.values()),
            return_exceptions=False,
        )


class CompositeMusicCatalog:
    """Route catalog operations without leaking provider branches into services."""

    def __init__(
        self,
        providers: MusicProviderRegistry,
        default_source: MusicSourceKind,
    ) -> None:
        self._providers = providers
        self._default_source = MusicSourceKind(default_source)
        providers.get(self._default_source)

    @property
    def default_source(self) -> MusicSourceKind:
        return self._default_source

    @property
    def sources(self) -> tuple[MusicSourceKind, ...]:
        return self._providers.sources

    async def search(
        self,
        query: str,
        *,
        limit: int,
        source: MusicSourceKind | None = None,
    ) -> tuple[MusicTrack, ...]:
        provider = self._providers.get(source or self._default_source)
        return await provider.search(query, limit=limit)

    async def lookup(self, reference: MusicTrackReference) -> MusicTrack:
        return await self._providers.get(reference.source).lookup(reference)

    async def inspect(self, locator: MusicPageLocator) -> MusicTrack:
        return await self._providers.get(locator.source).inspect(locator)

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        return await self._providers.get(track.source).resolve(track)

    async def health(self) -> tuple[MusicProviderHealth, ...]:
        return await self._providers.health()

    async def aclose(self) -> None:
        await self._providers.aclose()
