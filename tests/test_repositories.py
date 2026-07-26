from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.features.agent.models import ModelCapability, ProviderProtocol
from cywl_oopz.features.agent.repository import SqlAlchemyProviderCatalogRepository
from cywl_oopz.features.chat.models import (
    ChatMessage,
    ChatRole,
    ConversationKey,
    ConversationSession,
)
from cywl_oopz.features.chat.repository import SqlAlchemyConversationRepository
from cywl_oopz.storage.models import LlmModelRecord, LlmProviderRecord


class FailingSessionContext:
    async def __aenter__(self):
        raise SQLAlchemyError("contains a database diagnostic")

    async def __aexit__(self, *_: object) -> None:
        return None


class FailingSessionFactory:
    def __call__(self) -> FailingSessionContext:
        return FailingSessionContext()


@pytest.mark.asyncio
async def test_repository_maps_sqlalchemy_failure_to_safe_database_error() -> None:
    repository = SqlAlchemyConversationRepository(FailingSessionFactory())
    key = ConversationKey("channel", "area", "channel", "person")

    with pytest.raises(DatabaseError) as error:
        await repository.get(key)

    assert "database diagnostic" not in str(error.value)


def test_repository_round_trips_domain_session_to_orm_record() -> None:
    session = ConversationSession(
        key=ConversationKey("private", "", "", "person"),
        messages=(
            ChatMessage(ChatRole.USER, "hello"),
            ChatMessage(ChatRole.ASSISTANT, "hi"),
        ),
        selected_model="model",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    record = SqlAlchemyConversationRepository._to_record(session)
    round_tripped = SqlAlchemyConversationRepository._to_domain(record)

    assert round_tripped == session


def test_provider_catalog_repository_maps_credentials_and_capabilities() -> None:
    provider_id = uuid4()
    provider_record = LlmProviderRecord(
        id=provider_id,
        alias="local",
        display_name="Local gateway",
        protocol="openai_chat_compatible",
        base_url="https://llm.example/v1",
        api_key="stored-in-postgresql",
        user_selectable=True,
        enabled=True,
        config={"timeout_seconds": 10},
    )
    model_record = LlmModelRecord(
        id=uuid4(),
        provider_id=provider_id,
        alias="tool-model",
        remote_model_name="remote-tool-model",
        display_name="Tool model",
        enabled=True,
        is_provider_default=True,
        is_application_default=True,
        capabilities=["tool_calling", "streaming"],
        limits={"context_tokens": 32000},
        pricing={},
    )

    mapped_provider = SqlAlchemyProviderCatalogRepository._provider_to_domain(provider_record)
    mapped_model = SqlAlchemyProviderCatalogRepository._model_to_domain(model_record)

    assert mapped_provider.protocol is ProviderProtocol.OPENAI_CHAT_COMPATIBLE
    assert mapped_provider.api_key == "stored-in-postgresql"
    assert mapped_model.capabilities == {
        ModelCapability.TOOL_CALLING,
        ModelCapability.STREAMING,
    }
    assert mapped_model.limits["context_tokens"] == 32000
