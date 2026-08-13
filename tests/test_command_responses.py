from __future__ import annotations

import pytest

from cywl_oopz.commands.responses import (
    CommandMessage,
    CommandMessageBudget,
    CommandMessageTooLongError,
    MessageOverflowPolicy,
)
from cywl_oopz.integrations.oopz.command_responses import OopzCommandResponder


def test_message_budget_counts_utf16_units_used_by_oopz() -> None:
    assert CommandMessageBudget.units("abc初音") == 5
    assert CommandMessageBudget.units("🎵") == 2


def test_message_budget_paginates_on_semantic_line_boundaries() -> None:
    budget = CommandMessageBudget(16)
    text = "第一行内容\n第二行内容\n第三行内容"

    pages = budget.pages(CommandMessage(text, MessageOverflowPolicy.PAGINATE))

    assert len(pages) > 1
    assert all(budget.units(page) <= 16 for page in pages)
    assert "".join(pages).replace("\n", "") == text.replace("\n", "")


def test_message_budget_truncates_without_splitting_an_emoji() -> None:
    budget = CommandMessageBudget(16)

    page = budget.pages(CommandMessage("初音未来🎵" * 8, MessageOverflowPolicy.TRUNCATE))[0]

    assert page.endswith("…")
    assert budget.units(page) <= 16


def test_message_budget_rejects_atomic_overflow() -> None:
    budget = CommandMessageBudget(16)

    with pytest.raises(CommandMessageTooLongError):
        budget.pages(CommandMessage("x" * 17, MessageOverflowPolicy.REJECT))


class FakeContext:
    def __init__(self) -> None:
        self.replies: list[str] = []
        self.sent: list[str] = []

    async def reply(self, text: str) -> str:
        self.replies.append(text)
        return text

    async def send(self, text: str) -> str:
        self.sent.append(text)
        return text

    async def react(self, emoji: str) -> str:
        return emoji


@pytest.mark.asyncio
async def test_oopz_responder_emits_every_bounded_page() -> None:
    context = FakeContext()
    responder = OopzCommandResponder(context, CommandMessageBudget(16))

    results = await responder.reply("a" * 40)

    assert isinstance(results, tuple)
    assert results == tuple(context.replies)
    assert len(context.replies) == 3
    assert all(CommandMessageBudget.units(page) <= 16 for page in context.replies)
