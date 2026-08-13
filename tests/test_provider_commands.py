from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.features.agent.commands import (
    AgentModelCommand,
    ProviderCommand,
    SkillsCommand,
    ToolsCommand,
)
from cywl_oopz.features.agent.models import (
    AgentModelRef,
    ModelCatalogView,
    ModelSelection,
    ModelSelectionSource,
    ProviderProtocol,
    SelectableModel,
)
from cywl_oopz.features.agent.skills.models import (
    AgentSkillDiscovery,
    AgentSkillLibrary,
    AgentSkillShare,
    AgentSkillShareSummary,
    SkillAccessKind,
    SkillShareStatus,
)
from cywl_oopz.features.agent.tools.models import ToolEffect
from cywl_oopz.features.agent.tools.policy import AvailableTool
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor
from cywl_oopz.testing.commands import dispatch_command


class FakeProviderService:
    enabled = True

    def __init__(self) -> None:
        self.selections: list[tuple[str, str | None, bool]] = []
        self.model_selections: list[str] = []

    async def current_selection(self, key) -> ModelSelection:
        del key
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

    def _choices(self) -> tuple[SelectableModel, ...]:
        return (
            SelectableModel("primary", "Primary AI", "chat", "Chat", True),
            SelectableModel("primary", "Primary AI", "reasoning", "Reasoning", False),
            SelectableModel("secondary", "Secondary AI", "fast", "Fast", True),
        )

    async def model_catalog_view(self, key) -> ModelCatalogView:
        return ModelCatalogView(await self.current_selection(key), self._choices())

    async def select_model(self, key, choice: str) -> str:
        del key
        self.model_selections.append(choice)
        if choice == "missing":
            raise ValueError("not found")
        return choice if "/" in choice else f"primary/{choice}"

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

    async def available_tools(self, key) -> tuple[AvailableTool, ...]:
        del key
        return (
            AvailableTool(
                "get_agent_status",
                "查看 Agent 状态。",
                ToolEffect.READ,
            ),
            AvailableTool(
                "react_to_message",
                "添加表情。",
                ToolEffect.WRITE,
            ),
        )

    async def available_skills(self, key) -> tuple[AgentSkillDiscovery, ...]:
        del key
        return (
            AgentSkillDiscovery(
                id=uuid4(),
                name="web-research",
                display_name="网页研究",
                description="搜索并阅读关键来源。",
                version="1",
                revision=2,
                required_tools=frozenset({"search_web"}),
                access=SkillAccessKind.OWNED,
            ),
        )


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


async def execute(command, name: str, arguments: tuple[str, ...], context: FakeContext) -> None:
    router = CommandRouter("/")
    router.register_definition(command.definition())
    text = " ".join((f"/{name}", *arguments))
    message = SimpleNamespace(
        plain_text=text,
        text=text,
        content=text,
        sender_id="person",
        area="",
        channel="",
        message_id="message",
        mention_list=(),
    )
    context.event.message = message
    await dispatch_command(router, message, context)


@pytest.mark.asyncio
async def test_provider_command_lists_and_switches_thread_model() -> None:
    service = FakeProviderService()
    command = ProviderCommand(service, ChatTaskSupervisor())
    list_context = FakeContext()
    use_context = FakeContext()

    await execute(command, "provider", (), list_context)
    await execute(command, "provider", ("secondary", "fast"), use_context)

    assert "🎛️ **当前模型** primary/chat · 当前对话" in list_context.replies[0]
    assert "**primary** Primary AI：chat（默认）、reasoning" in list_context.replies[0]
    assert "/provider <Provider> [模型]" in list_context.replies[0]
    assert use_context.replies == ["✅ **当前对话模型** secondary/fast"]
    assert service.selections == [("secondary", "fast", False)]


@pytest.mark.asyncio
async def test_provider_command_keeps_old_use_syntax_and_explains_personal_default() -> None:
    service = FakeProviderService()
    command = ProviderCommand(service, ChatTaskSupervisor())
    old_context = FakeContext()
    default_context = FakeContext()

    await execute(command, "provider", ("use", "secondary", "fast"), old_context)
    await execute(command, "provider", ("default", "primary"), default_context)

    assert old_context.replies == ["✅ **当前对话模型** secondary/fast"]
    assert "**个人默认模型** primary/default" in default_context.replies[0]
    assert "当前对话的独立选择不会被覆盖" in default_context.replies[0]
    assert service.selections == [
        ("secondary", "fast", False),
        ("primary", None, True),
    ]


@pytest.mark.asyncio
async def test_agent_model_command_lists_current_provider_and_accepts_short_alias() -> None:
    service = FakeProviderService()
    command = AgentModelCommand(service, ChatTaskSupervisor())
    list_context = FakeContext()
    switch_context = FakeContext()

    await execute(command, "model", (), list_context)
    await execute(command, "model", ("reasoning",), switch_context)

    rendered = list_context.replies[0]
    assert "🎛️ **当前模型** primary/chat · 当前对话" in rendered
    assert "**primary 可用模型**" in rendered
    assert "**reasoning** · Reasoning" in rendered
    assert "secondary" not in rendered
    assert switch_context.replies == ["✅ **当前对话模型** primary/reasoning"]
    assert service.model_selections == ["reasoning"]


@pytest.mark.asyncio
async def test_agent_model_and_provider_commands_return_specific_not_found_help() -> None:
    service = FakeProviderService()
    model_context = FakeContext()
    provider_context = FakeContext()

    await execute(
        AgentModelCommand(service, ChatTaskSupervisor()),
        "model",
        ("missing",),
        model_context,
    )
    service.select_provider = _missing_provider
    await execute(
        ProviderCommand(service, ChatTaskSupervisor()),
        "provider",
        ("unknown",),
        provider_context,
    )

    assert "没有找到可选模型「missing」" in model_context.replies[0]
    assert "/model <Provider>/<模型>" in model_context.replies[0]
    assert "没有找到可选的 Provider/模型「unknown」" in provider_context.replies[0]
    assert "/provider <Provider> [模型]" in provider_context.replies[0]


async def _missing_provider(*args, **kwargs) -> str:
    del args, kwargs
    raise ValueError("not found")


@pytest.mark.asyncio
async def test_tools_command_lists_effects_without_internal_configuration() -> None:
    context = FakeContext()

    await execute(ToolsCommand(FakeProviderService()), "tools", (), context)

    assert "get_agent_status（只读）" in context.replies[0]
    assert "react_to_message（写操作）" in context.replies[0]


@pytest.mark.asyncio
async def test_skills_command_lists_safe_discovery_metadata_only() -> None:
    context = FakeContext()

    await execute(SkillsCommand(FakeProviderService()), "skills", (), context)

    assert "**网页研究** web-research · 我的 · v1" in context.replies[0]
    assert "搜索并阅读关键来源" in context.replies[0]
    assert "先搜索，再阅读" not in context.replies[0]


@pytest.mark.asyncio
async def test_skills_command_lists_shared_skills_and_pending_invitations() -> None:
    shared = AgentSkillDiscovery(
        id=uuid4(),
        name="travel-planner",
        display_name="旅行规划",
        description="规划旅行时使用。",
        version="1",
        revision=1,
        required_tools=frozenset(),
        access=SkillAccessKind.SHARED,
    )
    now = datetime.now(UTC)
    invitation = AgentSkillShareSummary(
        AgentSkillShare(
            id=uuid4(),
            skill_id=shared.id,
            recipient_person_id="person",
            status=SkillShareStatus.PENDING,
            created_at=now,
            updated_at=now,
        ),
        shared,
    )

    class Library:
        async def library(self, person_id):
            assert person_id == "person"
            return AgentSkillLibrary(
                owned=(),
                builtin=(),
                shared=(shared,),
                pending_invitations=(invitation,),
            )

    command = SkillsCommand(FakeProviderService(), Library())
    shared_context = FakeContext()
    invitation_context = FakeContext()

    await execute(command, "skills", ("shared",), shared_context)
    await execute(command, "skills", ("invitations",), invitation_context)

    assert "**旅行规划** travel-planner · v1" in shared_context.replies[0]
    assert str(invitation.share.id) in invitation_context.replies[0]
    assert "person" not in invitation_context.replies[0]
