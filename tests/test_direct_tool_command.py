from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.core.lifecycle import ModelSelectionSource, ToolEffect
from cywl_oopz.features.agent.commands import ToolCommand
from cywl_oopz.features.agent.direct_tools import DirectToolService
from cywl_oopz.features.agent.models import (
    AgentIdentity,
    AgentModelRef,
    ModelCapability,
    ModelSelection,
    ProviderProtocol,
)
from cywl_oopz.features.agent.skills.tools import LoadAgentSkillTool
from cywl_oopz.features.agent.tools.models import (
    ToolDescriptor,
    ToolExecutionContext,
)
from cywl_oopz.features.agent.tools.registry import ToolRegistry
from cywl_oopz.settings import AgentSettings
from cywl_oopz.testing.commands import dispatch_command


class EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class EchoOutput(BaseModel):
    value: str


class EchoTool:
    descriptor = ToolDescriptor(
        name="echo_debug",
        display_name="Echo debug",
        description="Return the supplied text.",
        input_model=EchoInput,
        output_model=EchoOutput,
        effect=ToolEffect.READ,
        timeout_seconds=1,
        max_output_characters=1000,
        concurrency_safe=True,
        idempotent=True,
    )

    def __init__(self) -> None:
        self.received = ""

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        del context
        self.received = cast(EchoInput, arguments).value
        return EchoOutput(value=self.received)


class FakeSelectionService:
    async def resolve(
        self,
        key,
        *,
        required_capabilities=frozenset(),
    ) -> ModelSelection:
        del key
        assert required_capabilities == frozenset({ModelCapability.TOOL_CALLING})
        return ModelSelection(
            AgentModelRef(
                provider_id=uuid4(),
                model_id=uuid4(),
                provider_alias="provider",
                model_alias="model",
                remote_model_name="model",
                protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
                capabilities=frozenset({ModelCapability.TOOL_CALLING}),
                fallback_model_id=None,
            ),
            ModelSelectionSource.APPLICATION,
        )


class FakeAvailabilityService:
    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names
        self.identity: AgentIdentity | None = None

    async def names(
        self,
        identity: AgentIdentity,
        model: AgentModelRef,
    ) -> tuple[str, ...]:
        del model
        self.identity = identity
        return self._names


@dataclass
class FakeMessage:
    plain_text: str
    sender_id: str = "person"
    area: str = "area"
    channel: str = "channel"
    message_id: str = "message"
    text: str = ""
    content: str = ""
    mention_list: tuple[object, ...] = ()


class FakeContext:
    def __init__(self) -> None:
        message = SimpleNamespace(
            sender_id="person",
            area="area",
            channel="channel",
            message_id="message",
        )
        self.event = SimpleNamespace(is_private=False, message=message)
        self.replies: list[str] = []

    async def reply(self, content: str) -> None:
        self.replies.append(content)


def direct_command(
    *,
    enabled_tools: tuple[str, ...] = ("echo_debug",),
) -> tuple[CommandRouter, EchoTool, FakeAvailabilityService]:
    tool = EchoTool()
    availability = FakeAvailabilityService(enabled_tools)
    service = DirectToolService(
        AgentSettings.from_mapping({"CYWL_AGENT_MODE": "agent"}),
        ToolRegistry((tool,)),
        availability,  # type: ignore[arg-type]
        FakeSelectionService(),  # type: ignore[arg-type]
    )
    router = CommandRouter("/")
    router.register_definition(ToolCommand(service, "/").definition())
    return router, tool, availability


@pytest.mark.asyncio
async def test_tool_command_executes_json_without_losing_string_spacing() -> None:
    router, tool, availability = direct_command()
    context = FakeContext()

    consumed = await dispatch_command(
        router,
        FakeMessage('/tool echo_debug {"value": "hello  world"}'),  # type: ignore[arg-type]
        context,  # type: ignore[arg-type]
    )

    assert consumed is True
    assert json.loads(context.replies[-1]) == {
        "ok": True,
        "data": {"value": "hello  world"},
    }
    assert tool.received == "hello  world"
    assert availability.identity is not None
    assert availability.identity.source_message_id == "message"
    assert availability.identity.transport_channel_id == "channel"


@pytest.mark.asyncio
async def test_tool_command_help_returns_descriptor_and_schemas() -> None:
    router, _, _ = direct_command()
    context = FakeContext()

    await dispatch_command(
        router,
        FakeMessage("/tool echo_debug --help"),  # type: ignore[arg-type]
        context,  # type: ignore[arg-type]
    )

    payload = json.loads(context.replies[-1])
    assert payload["ok"] is True
    assert payload["data"]["id"] == "echo_debug"
    assert payload["data"]["effect"] == "read"
    assert payload["data"]["input_schema"]["properties"]["value"]["type"] == "string"
    assert payload["data"]["output_schema"]["properties"]["value"]["type"] == "string"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "error"),
    [
        ("/tool echo_debug nope", "invalid_json"),
        ("/tool echo_debug []", "json_body_must_be_object"),
        ("/tool echo_debug {}", "invalid_arguments"),
        ("/tool missing --help", "tool_not_registered"),
    ],
)
async def test_tool_command_returns_json_errors(command: str, error: str) -> None:
    router, _, _ = direct_command()
    context = FakeContext()

    await dispatch_command(
        router,
        FakeMessage(command),  # type: ignore[arg-type]
        context,  # type: ignore[arg-type]
    )

    assert json.loads(context.replies[-1])["error"] == error


@pytest.mark.asyncio
async def test_tool_command_honors_tool_availability() -> None:
    router, tool, _ = direct_command(enabled_tools=())
    context = FakeContext()

    await dispatch_command(
        router,
        FakeMessage('/tool echo_debug {"value":"hello"}'),  # type: ignore[arg-type]
        context,  # type: ignore[arg-type]
    )

    assert json.loads(context.replies[-1]) == {
        "ok": False,
        "error": "tool_not_enabled",
    }
    assert tool.received == ""


@pytest.mark.asyncio
async def test_direct_skill_loader_is_explicitly_unavailable_without_run_scope() -> None:
    tool = LoadAgentSkillTool()
    availability = FakeAvailabilityService(("load_agent_skill",))
    service = DirectToolService(
        AgentSettings.from_mapping({"CYWL_AGENT_MODE": "agent"}),
        ToolRegistry((tool,)),
        availability,  # type: ignore[arg-type]
        FakeSelectionService(),  # type: ignore[arg-type]
    )
    router = CommandRouter("/")
    router.register_definition(ToolCommand(service, "/").definition())
    context = FakeContext()

    await dispatch_command(
        router,
        FakeMessage(f'/tool load_agent_skill {{"skill_id":"{uuid4()}"}}'),  # type: ignore[arg-type]
        context,  # type: ignore[arg-type]
    )

    assert json.loads(context.replies[-1]) == {
        "ok": False,
        "error": "skill_catalog_unavailable",
    }


def test_tool_command_always_renders_valid_bounded_json() -> None:
    rendered = ToolCommand._render({"ok": True, "data": {"value": "x" * 3000}})

    assert len(rendered) <= ToolCommand.max_reply_characters
    assert json.loads(rendered)["truncated"] is True
