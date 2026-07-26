from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import SQLAlchemyError

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.features.chat.models import (
    ChatMessage,
    ChatRole,
    ConversationKey,
    ConversationSession,
)
from cywl_oopz.features.chat.repository import SqlAlchemyConversationRepository


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
