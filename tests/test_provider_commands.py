from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cywl_oopz.commands.router import ParsedCommand
from cywl_oopz.features.agent.commands import ProviderCommand
from cywl_oopz.features.agent.models import (
    AgentModelRef,
    ModelSelection,
    ModelSelectionSource,
    ProviderProtocol,
)
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor


class FakeProviderService:
    enabled = True

    def __init__(self) -> None:
        self.selections: list[tuple[str, str | None, bool]] = []

    async def current_selection(self, key) -> ModelSelection:
        return ModelSelection(
            AgentModelRef(
                provider_id=uuid4(),
                model_id=uuid4(),
                provider_alias="primary",
                model_alias="chat",
                remote_model_name="remote",
                protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
                capabilities=frozenset(),
                fallback_model_id=None,
            ),
            ModelSelectionSource.THREAD,
        )

    def list_models(self) -> tuple[str, ...]:
        return ("primary/chat", "secondary/fast")

    async def select_provider(
        self,
        key,
        provider_alias: str,
        model_alias: str | None,
        *,
        user_default: bool,
    ) -> str:
        self.selections.append((provider_alias, model_alias, user_default))
        return f"{provider_alias}/{model_alias or 'default'}"


@dataclass
class FakeContext:
    replies: list[str] = field(default_factory=list)
    event: object = field(
        default_factory=lambda: SimpleNamespace(
            is_private=True,
            message=SimpleNamespace(sender_id="person"),
        )
    )

    async def reply(self, text: str) -> None:
        self.replies.append(text)


@pytest.mark.asyncio
async def test_provider_command_lists_and_switches_thread_model() -> None:
    service = FakeProviderService()
    command = ProviderCommand(service, ChatTaskSupervisor())
    list_context = FakeContext()
    use_context = FakeContext()

    await command.execute(ParsedCommand("provider", ("list",)), list_context)
    await command.execute(
        ParsedCommand("provider", ("use", "secondary", "fast")),
        use_context,
    )

    assert "primary/chat" in list_context.replies[0]
    assert use_context.replies == ["当前对话模型已切换为：secondary/fast"]
    assert service.selections == [("secondary", "fast", False)]
