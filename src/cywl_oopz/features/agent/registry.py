"""Pydantic AI model construction and asynchronous HTTP client ownership."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from cywl_oopz.core.errors import ConfigurationError, ProviderSelectionError

from .catalog import ReloadableProviderCatalog
from .models import AgentModelRef, LlmProvider, ProviderProtocol
from .provider_retry import current_provider_retry_progress

logger = logging.getLogger(__name__)


class ObservableAsyncOpenAI(AsyncOpenAI):
    """Expose the SDK's request-level exponential backoff as safe progress."""

    def __init__(
        self,
        *,
        provider_alias: str,
        retry_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        **kwargs: Any,
    ) -> None:
        self._cywl_provider_alias = provider_alias
        self._cywl_retry_sleep = retry_sleep
        super().__init__(**kwargs)

    async def _sleep_for_retry(
        self,
        *,
        retries_taken: int,
        max_retries: int,
        options: Any,
        response: httpx.Response | None,
    ) -> None:
        """Mirror the SDK hook while adding logging and run-scoped progress."""
        remaining_retries = max_retries - retries_taken
        delay = self._calculate_retry_timeout(
            remaining_retries,
            options,
            response.headers if response is not None else None,
        )
        retry_attempt = retries_taken + 1
        status = response.status_code if response is not None else None
        reason = self._retry_reason(status)
        logger.warning(
            "Agent provider request retry scheduled: provider=%s retry=%s/%s "
            "delay_seconds=%.2f status=%s",
            self._cywl_provider_alias,
            retry_attempt,
            max_retries,
            delay,
            status if status is not None else "transport",
        )
        progress = current_provider_retry_progress()
        if progress is not None:
            await progress.waiting(
                attempt=retry_attempt,
                max_attempts=max_retries,
                delay_seconds=delay,
                reason=reason,
            )
        await self._cywl_retry_sleep(delay)
        if progress is not None:
            await progress.resumed()

    @staticmethod
    def _retry_reason(status: int | None) -> str:
        if status is None:
            return "网络连接超时或中断"
        if status == 408:
            return "上游请求超时（HTTP 408）"
        if status == 409:
            return "上游请求暂时冲突（HTTP 409）"
        if status == 429:
            return "上游服务繁忙（HTTP 429）"
        if status >= 500:
            return f"上游服务异常（HTTP {status}）"
        return f"上游请求暂时失败（HTTP {status}）"


@dataclass(slots=True)
class _ClientEntry:
    fingerprint: tuple[object, ...]
    client: httpx.AsyncClient


class ProviderClientPool:
    """Cache one async client per unchanged provider configuration."""

    def __init__(self) -> None:
        self._entries: dict[UUID, _ClientEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, provider: LlmProvider) -> httpx.AsyncClient:
        """Return a client, replacing stale configuration under the same stable ID."""
        fingerprint = self._fingerprint(provider)
        async with self._lock:
            existing = self._entries.get(provider.id)
            if existing is not None and existing.fingerprint == fingerprint:
                logger.debug("Reusing Agent provider HTTP client: provider=%s", provider.alias)
                return existing.client

            client = httpx.AsyncClient(
                timeout=self._timeout(provider.config),
                headers=self._headers(provider.config),
            )
            self._entries[provider.id] = _ClientEntry(fingerprint, client)
            if existing is not None:
                logger.info("Replacing Agent provider HTTP client: provider=%s", provider.alias)
                await existing.client.aclose()
            else:
                logger.info("Created Agent provider HTTP client: provider=%s", provider.alias)
            return client

    async def aclose(self) -> None:
        """Close and forget all clients before the database and event loop close."""
        async with self._lock:
            entries = tuple(self._entries.values())
            self._entries.clear()
        if entries:
            logger.info("Closing Agent provider HTTP clients: count=%s", len(entries))
            await asyncio.gather(
                *(entry.client.aclose() for entry in entries),
                return_exceptions=True,
            )

    @staticmethod
    def _fingerprint(provider: LlmProvider) -> tuple[object, ...]:
        return (
            provider.base_url,
            provider.api_key,
            repr(sorted(provider.config.items())),
        )

    @staticmethod
    def _timeout(config: Mapping[str, Any]) -> float:
        raw = config.get("timeout_seconds", 45.0)
        if not isinstance(raw, int | float) or raw <= 0:
            raise ConfigurationError("Provider timeout_seconds must be positive")
        return float(raw)

    @staticmethod
    def _headers(config: Mapping[str, Any]) -> dict[str, str]:
        raw = config.get("headers", {})
        if not isinstance(raw, dict):
            raise ConfigurationError("Provider headers must be a JSON object")
        return {str(name): str(value) for name, value in raw.items()}


class AgentModelRegistry:
    """Map project model references to framework models without leaking credentials."""

    def __init__(
        self,
        catalog: ReloadableProviderCatalog,
        clients: ProviderClientPool | None = None,
        *,
        default_max_retries: int = 2,
    ) -> None:
        if not 0 <= default_max_retries <= 5:
            raise ValueError("Default provider retries must be between 0 and 5")
        self._catalog = catalog
        self._clients = clients or ProviderClientPool()
        self._default_max_retries = default_max_retries

    async def reload(self) -> None:
        """Refresh the catalog; stale clients are replaced lazily by fingerprint."""
        logger.debug("Reloading Agent model registry catalog")
        await self._catalog.reload()

    async def model(self, reference: AgentModelRef) -> OpenAIChatModel:
        """Build a lightweight Pydantic AI model over a pooled provider client."""
        snapshot = self._catalog.snapshot
        provider = snapshot.providers.get(reference.provider_id)
        model = snapshot.models.get(reference.model_id)
        if (
            provider is None
            or model is None
            or model.provider_id != provider.id
            or model.remote_model_name != reference.remote_model_name
        ):
            logger.warning(
                "Pinned Agent model disappeared from catalog: provider=%s model=%s",
                reference.provider_alias,
                reference.model_alias,
            )
            raise ProviderSelectionError("Pinned Agent model is no longer in the catalog")
        if provider.protocol is not ProviderProtocol.OPENAI_CHAT_COMPATIBLE:
            logger.warning(
                "Unsupported Agent provider protocol: provider=%s protocol=%s",
                provider.alias,
                provider.protocol.value,
            )
            raise ConfigurationError(f"Unsupported provider protocol: {provider.protocol}")

        client = await self._clients.get(provider)
        openai_client = ObservableAsyncOpenAI(
            provider_alias=provider.alias,
            base_url=provider.base_url,
            api_key=provider.api_key,
            http_client=client,
            max_retries=self._max_retries(provider.config),
        )
        pydantic_provider = OpenAIProvider(
            openai_client=openai_client,
        )
        profile = provider.config.get("profile")
        if profile is not None and not isinstance(profile, dict):
            raise ConfigurationError("Provider profile must be a JSON object")
        return OpenAIChatModel(
            model.remote_model_name,
            provider=pydantic_provider,
            profile=profile,
        )

    async def aclose(self) -> None:
        """Close all provider clients."""
        logger.debug("Closing Agent model registry")
        await self._clients.aclose()

    def _max_retries(self, config: Mapping[str, Any]) -> int:
        raw = config.get("max_retries", self._default_max_retries)
        if type(raw) is not int or not 0 <= raw <= 10:
            raise ConfigurationError("Provider max_retries must be an integer between 0 and 10")
        return raw
