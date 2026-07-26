from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from cywl_oopz.features.agent.models import LlmProvider, ProviderProtocol
from cywl_oopz.features.agent.registry import ProviderClientPool


def provider(api_key: str = "first-key") -> LlmProvider:
    return LlmProvider(
        id=uuid4(),
        alias="provider",
        display_name="Provider",
        protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
        base_url="https://llm.example/v1",
        api_key=api_key,
        user_selectable=True,
        enabled=True,
        config={"timeout_seconds": 5, "headers": {"X-Client": "cywl"}},
    )


@pytest.mark.asyncio
async def test_client_pool_reuses_unchanged_provider_and_replaces_changed_credentials() -> None:
    pool = ProviderClientPool()
    original_provider = provider()

    first = await pool.get(original_provider)
    reused = await pool.get(original_provider)
    replacement = await pool.get(replace(original_provider, api_key="second-key"))

    assert first is reused
    assert replacement is not first
    assert first.is_closed is True
    assert replacement.headers["X-Client"] == "cywl"

    await pool.aclose()
    assert replacement.is_closed is True
