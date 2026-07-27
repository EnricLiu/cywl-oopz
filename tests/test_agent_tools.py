from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from cywl_oopz.features.agent.models import (
    AgentIdentity,
    AgentModelRef,
    AgentRunLimits,
    ModelCapability,
    ProviderProtocol,
)
from cywl_oopz.features.agent.tools.builtin import (
    ReactToMessageInput,
    ReactToMessageTool,
)
from cywl_oopz.features.agent.tools.executor import ToolExecutor
from cywl_oopz.features.agent.tools.models import (
    ToolCall,
    ToolDescriptor,
    ToolEffect,
    ToolExecution,
    ToolExecutionClaim,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionStatus,
)
from cywl_oopz.features.agent.tools.policy import (
    ToolAvailabilityService,
    ToolPolicy,
)
from cywl_oopz.features.agent.tools.registry import ToolRegistry
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.integrations.oopz.reactions import OopzReactionGateway


class NumberInput(BaseModel):
    value: int


class NumberOutput(BaseModel):
    value: int


class RecordingTool:
    def __init__(
        self,
        name: str = "double",
        *,
        effect: ToolEffect = ToolEffect.READ,
        wait: asyncio.Event | None = None,
        timeout_seconds: float = 1,
    ) -> None:
        self._descriptor = ToolDescriptor(
            name=name,
            display_name=name,
            description="Double one integer.",
            input_model=NumberInput,
            output_model=NumberOutput,
            effect=effect,
            timeout_seconds=timeout_seconds,
            max_output_characters=1000,
            concurrency_safe=effect is ToolEffect.READ,
            idempotent=True,
        )
        self.wait = wait
        self.calls = 0

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        del context
        self.calls += 1
        if self.wait is not None:
            await self.wait.wait()
        value = cast(NumberInput, arguments).value
        return NumberOutput(value=value * 2)


class InMemoryExecutionRepository:
    def __init__(self) -> None:
        self.records: dict[tuple[UUID, str], ToolExecution] = {}

    async def claim(self, execution: ToolExecution) -> ToolExecutionClaim:
        key = (execution.run_id, execution.call_id)
        existing = self.records.get(key)
        if existing is None:
            existing = next(
                (
                    item
                    for item in self.records.values()
                    if item.run_id == execution.run_id
                    and item.idempotency_key == execution.idempotency_key
                ),
                None,
            )
        if existing is not None:
            return ToolExecutionClaim(existing, created=False)
        self.records[key] = execution
        return ToolExecutionClaim(execution, created=True)

    async def finish(
        self,
        run_id: UUID,
        call_id: str,
        status: ToolExecutionStatus,
        *,
        output: dict[str, object] | None,
        error_code: str,
    ) -> ToolExecution:
        key = (run_id, call_id)
        execution = replace(
            self.records[key],
            status=status,
            output_payload=output,
            error_code=error_code,
            finished_at=datetime.now(UTC),
        )
        self.records[key] = execution
        return execution


class FakeChannels:
    def __init__(self, enabled: frozenset[str]) -> None:
        self.enabled = enabled

    async def is_chat_enabled(self, area_id: str, channel_id: str) -> bool:
        del area_id, channel_id
        return True

    async def enabled_agent_tools(
        self,
        area_id: str,
        channel_id: str,
    ) -> frozenset[str]:
        del area_id, channel_id
        return self.enabled


def tool_context(*enabled: str) -> ToolExecutionContext:
    key = ConversationKey("channel", "area", "channel", "person")
    return ToolExecutionContext(
        run_id=uuid4(),
        identity=AgentIdentity("person", key),
        limits=AgentRunLimits(),
        enabled_tools=tuple(enabled),
    )


def tool_model(*capabilities: ModelCapability) -> AgentModelRef:
    return AgentModelRef(
        provider_id=uuid4(),
        model_id=uuid4(),
        provider_alias="provider",
        model_alias="model",
        remote_model_name="model",
        protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
        capabilities=frozenset(capabilities),
        fallback_model_id=None,
    )


def test_write_tools_must_declare_idempotency_and_registry_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="idempotency"):
        ToolDescriptor(
            name="write",
            display_name="write",
            description="write",
            input_model=NumberInput,
            output_model=NumberOutput,
            effect=ToolEffect.WRITE,
        )

    first = RecordingTool()
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry((first, RecordingTool()))

    bounded = ToolExecutor._bounded_output({"text": "\\" * 1000}, 128)
    assert bounded["truncated"] is True
    assert len(json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))) <= 128


@pytest.mark.asyncio
async def test_executor_validates_policy_and_replays_success_without_side_effect() -> None:
    tool = RecordingTool()
    write = RecordingTool("write_value", effect=ToolEffect.WRITE)
    repository = InMemoryExecutionRepository()
    executor = ToolExecutor(ToolRegistry((tool, write)), ToolPolicy(), repository)
    context = tool_context("double")
    call = ToolCall("call-1", "double", {"value": 3})

    first = await executor.execute(call, context)
    replay = await executor.execute(call, context)
    invalid = await executor.execute(
        ToolCall("call-invalid", "double", {"value": "nope"}),
        context,
    )
    denied = await executor.execute(
        ToolCall("call-denied", "double", {"value": 4}),
        replace(context, enabled_tools=()),
    )
    write_context = replace(context, enabled_tools=("write_value",))
    first_write = await executor.execute(
        ToolCall("write-1", "write_value", {"value": 5}),
        write_context,
    )
    repeated_write = await executor.execute(
        ToolCall("write-2", "write_value", {"value": 5}),
        write_context,
    )

    assert first.model_payload() == {"ok": True, "data": {"value": 6}}
    assert replay.model_payload() == first.model_payload()
    assert tool.calls == 1
    assert invalid.error_code == "invalid_arguments"
    assert invalid.status is ToolExecutionStatus.FAILED
    assert denied.error_code == "tool_not_enabled"
    assert denied.status is ToolExecutionStatus.DENIED
    assert repeated_write.model_payload() == first_write.model_payload()
    assert write.calls == 1


@pytest.mark.asyncio
async def test_executor_propagates_cancellation_and_persists_terminal_state() -> None:
    wait = asyncio.Event()
    tool = RecordingTool(wait=wait)
    repository = InMemoryExecutionRepository()
    executor = ToolExecutor(ToolRegistry((tool,)), ToolPolicy(), repository)
    context = tool_context("double")
    task = asyncio.create_task(
        executor.execute(ToolCall("call-cancel", "double", {"value": 2}), context)
    )
    while tool.calls == 0:
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    execution = repository.records[(context.run_id, "call-cancel")]
    assert execution.status is ToolExecutionStatus.CANCELLED
    assert execution.error_code == "cancelled"


@pytest.mark.asyncio
async def test_executor_bounds_timeout_and_denies_admin_for_normal_identity() -> None:
    wait = asyncio.Event()
    slow = RecordingTool("slow", wait=wait, timeout_seconds=0.001)
    admin = RecordingTool("admin_tool", effect=ToolEffect.ADMIN)
    repository = InMemoryExecutionRepository()
    executor = ToolExecutor(
        ToolRegistry((slow, admin)),
        ToolPolicy(),
        repository,
    )

    timed_out = await executor.execute(
        ToolCall("call-timeout", "slow", {"value": 1}),
        tool_context("slow"),
    )
    denied = await executor.execute(
        ToolCall("call-admin", "admin_tool", {"value": 1}),
        tool_context("admin_tool"),
    )

    assert timed_out.error_code == "tool_timeout"
    assert denied.error_code == "administrator_required"
    assert admin.calls == 0


@pytest.mark.asyncio
async def test_executor_preserves_expected_tool_error_code() -> None:
    class ExpectedFailureTool(RecordingTool):
        async def execute(
            self,
            context: ToolExecutionContext,
            arguments: BaseModel,
        ) -> BaseModel:
            del context, arguments
            raise ToolExecutionError("voice_channel_required")

    tool = ExpectedFailureTool("expected_failure")
    executor = ToolExecutor(
        ToolRegistry((tool,)),
        ToolPolicy(),
        InMemoryExecutionRepository(),
    )

    result = await executor.execute(
        ToolCall("call-expected", "expected_failure", {"value": 1}),
        tool_context("expected_failure"),
    )

    assert result.status is ToolExecutionStatus.FAILED
    assert result.error_code == "voice_channel_required"


@pytest.mark.asyncio
async def test_availability_intersects_channel_and_model_capability() -> None:
    read = RecordingTool("read_tool")
    write = RecordingTool("write_tool", effect=ToolEffect.WRITE)
    admin = RecordingTool("admin_tool", effect=ToolEffect.ADMIN)
    registry = ToolRegistry((read, write, admin))
    availability = ToolAvailabilityService(
        registry,
        FakeChannels(frozenset({"read_tool", "admin_tool"})),
        ("read_tool", "write_tool", "admin_tool"),
    )
    identity = AgentIdentity(
        "person",
        ConversationKey("channel", "area", "channel", "person"),
    )

    enabled = await availability.resolve(
        identity,
        tool_model(ModelCapability.TOOL_CALLING),
    )
    unsupported = await availability.resolve(identity, tool_model())
    administrator_enabled = await availability.resolve(
        replace(identity, is_administrator=True),
        tool_model(ModelCapability.TOOL_CALLING),
    )

    assert [item.name for item in enabled] == ["read_tool"]
    assert unsupported == ()
    assert [item.name for item in administrator_enabled] == [
        "admin_tool",
        "read_tool",
    ]


@pytest.mark.asyncio
async def test_reaction_tool_uses_trusted_identity_not_model_arguments() -> None:
    class FakeReactions:
        def __init__(self) -> None:
            self.calls: list[tuple[AgentIdentity, str]] = []

        async def add_reaction(self, identity: AgentIdentity, emoji: str) -> None:
            self.calls.append((identity, emoji))

    gateway = FakeReactions()
    tool = ReactToMessageTool(
        gateway,
        timeout_seconds=1,
        max_output_characters=1000,
    )
    identity = AgentIdentity(
        "person",
        ConversationKey("channel", "area", "channel", "person"),
        source_message_id="message",
        transport_channel_id="channel",
    )
    context = replace(tool_context("react_to_message"), identity=identity)

    output = await tool.execute(context, ReactToMessageInput(emoji="🎉"))

    assert output.model_dump() == {"applied": True, "emoji": "🎉"}
    assert gateway.calls == [(identity, "🎉")]


@pytest.mark.asyncio
async def test_oopz_reaction_gateway_maps_channel_and_private_targets() -> None:
    class FakeMessages:
        def __init__(self) -> None:
            self.channel_calls: list[dict[str, str]] = []
            self.private_calls: list[dict[str, str]] = []

        async def add_channel_reaction(self, **values: str) -> None:
            self.channel_calls.append(values)

        async def add_private_reaction(self, **values: str) -> None:
            self.private_calls.append(values)

    class FakeBot:
        def __init__(self) -> None:
            self.messages = FakeMessages()

    bot = FakeBot()
    gateway = OopzReactionGateway(bot)
    channel_identity = AgentIdentity(
        "person",
        ConversationKey("channel", "area", "channel", "person"),
        source_message_id="channel-message",
        transport_channel_id="channel",
    )
    private_identity = AgentIdentity(
        "person",
        ConversationKey("private", "", "", "person"),
        source_message_id="private-message",
        transport_channel_id="private-session",
    )

    await gateway.add_reaction(channel_identity, "👍")
    await gateway.add_reaction(private_identity, "❤️")

    assert bot.messages.channel_calls == [
        {
            "message_id": "channel-message",
            "area": "area",
            "channel": "channel",
            "emoji": "👍",
        }
    ]
    assert bot.messages.private_calls == [
        {
            "message_id": "private-message",
            "channel": "private-session",
            "target": "person",
            "emoji": "❤️",
        }
    ]
