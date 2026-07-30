from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from cywl_oopz.features.agent.skills.models import AgentSkillDiscovery, SkillAccessKind
from cywl_oopz.integrations.oopz.skill_sharing import OopzSkillShareNotifier


def discovery() -> AgentSkillDiscovery:
    return AgentSkillDiscovery(
        id=uuid4(),
        name="travel-planner",
        display_name="旅行规划",
        description="规划旅行时使用。",
        version="1",
        revision=1,
        required_tools=frozenset(),
        access=SkillAccessKind.OWNED,
    )


@pytest.mark.asyncio
async def test_skill_share_notifier_sends_compact_private_invitation() -> None:
    sent: list[tuple[tuple[str, ...], dict[str, str]]] = []

    class Messages:
        async def send_private_message(self, *texts: str, **kwargs: str) -> None:
            sent.append((texts, kwargs))

    notifier = OopzSkillShareNotifier(SimpleNamespace(messages=Messages()))

    assert await notifier.invitation("friend", discovery()) is True
    assert sent[0][1] == {"target": "friend"}
    assert "旅行规划" in sent[0][0][0]
    assert "friend" not in sent[0][0][0]


@pytest.mark.asyncio
async def test_skill_share_notifier_reports_failure_without_raising() -> None:
    class Messages:
        async def send_private_message(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("offline")

    notifier = OopzSkillShareNotifier(SimpleNamespace(messages=Messages()))

    assert await notifier.revoked("friend", discovery()) is False
