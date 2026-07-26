from __future__ import annotations

import pytest

from cywl_oopz.settings import ChatSettings


@pytest.fixture
def chat_settings() -> ChatSettings:
    return ChatSettings(
        enabled=True,
        base_url="https://llm.example/v1",
        api_key="test-key",
        model="test-model",
        allowed_models=("test-model", "alternate-model"),
        model_selection_users=frozenset({"model-admin"}),
        system_prompt="You are a test assistant.",
        request_timeout_seconds=5.0,
        stream_responses=True,
        session_ttl_seconds=3600,
        max_history_messages=6,
        max_history_characters=100,
        user_cooldown_seconds=0.0,
        max_global_concurrency=4,
        max_channel_concurrency=2,
        max_user_concurrency=1,
    )
