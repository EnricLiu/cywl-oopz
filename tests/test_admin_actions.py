from __future__ import annotations

from dataclasses import dataclass

import pytest

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.features.admin.actions import (
    DebugActionStatus,
    DebugMessageAction,
    MessageActionTarget,
    RecallActionStatus,
    RecallMessageAction,
)
from cywl_oopz.features.admin.models import (
    MessageRecallOutcome,
    OopzMessageAddress,
    OopzMessageScope,
)
from cywl_oopz.features.admin.recall import (
    BotMessageRecallTransportError,
    ReferencedBotMessageNotFoundError,
)

ADDRESS = OopzMessageAddress(OopzMessageScope.CHANNEL, "area", "channel")
TARGET = MessageActionTarget("message", ADDRESS)


class RecallService:
    def __init__(self, value: object) -> None:
        self.value = value

    async def recall(self, message_id, address, embedded=None):
        del message_id, address, embedded
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (MessageRecallOutcome.RECALLED, RecallActionStatus.RECALLED),
        (MessageRecallOutcome.ALREADY_RECALLED, RecallActionStatus.ALREADY_RECALLED),
        (ReferencedBotMessageNotFoundError(), RecallActionStatus.NOT_APPLICABLE),
        (BotMessageRecallTransportError(), RecallActionStatus.UNAVAILABLE),
        (DatabaseError("unavailable"), RecallActionStatus.PERSISTENCE_UNAVAILABLE),
    ],
)
async def test_recall_action_maps_trigger_neutral_outcomes(value, expected) -> None:
    action = RecallMessageAction(RecallService(value))  # type: ignore[arg-type]

    assert await action.execute(TARGET) is expected


@dataclass
class DiagnosticRepository:
    value: object

    async def get_by_outbound_message(self, message_id, address):
        del message_id, address
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class DiagnosticRenderer:
    def render(self, diagnostic, *, verbose):
        return (f"diagnostic={diagnostic} verbose={verbose}",)


@pytest.mark.asyncio
async def test_debug_action_distinguishes_completed_missing_and_unavailable() -> None:
    completed = DebugMessageAction(
        DiagnosticRepository("run"),  # type: ignore[arg-type]
        DiagnosticRenderer(),  # type: ignore[arg-type]
    )
    missing = DebugMessageAction(
        DiagnosticRepository(None),  # type: ignore[arg-type]
        DiagnosticRenderer(),  # type: ignore[arg-type]
    )
    unavailable = DebugMessageAction(
        DiagnosticRepository(DatabaseError("unavailable")),  # type: ignore[arg-type]
        DiagnosticRenderer(),  # type: ignore[arg-type]
    )

    completed_result = await completed.execute(TARGET, verbose=True)

    assert completed_result.status is DebugActionStatus.COMPLETED
    assert completed_result.pages == ("diagnostic=run verbose=True",)
    assert (await missing.execute(TARGET, verbose=False)).status is (
        DebugActionStatus.NOT_APPLICABLE
    )
    assert (await unavailable.execute(TARGET, verbose=False)).status is (
        DebugActionStatus.UNAVAILABLE
    )
