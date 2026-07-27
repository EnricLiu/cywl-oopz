"""Small internal tools suitable for the first Agent tool allow-list."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from cywl_oopz.features.agent.models import AgentIdentity

from .models import ToolDescriptor, ToolEffect, ToolExecutionContext
from .policy import ChannelToolSettings


class ReactionGateway(Protocol):
    """OOPZ-independent boundary for reacting to the invocation message."""

    async def add_reaction(self, identity: AgentIdentity, emoji: str) -> None:
        """Apply one allowed reaction to the source message."""


class EmptyToolInput(BaseModel):
    """Input for tools that need only trusted execution context."""

    model_config = ConfigDict(extra="forbid")


class AgentStatusOutput(BaseModel):
    """Safe snapshot of the current bounded Agent run."""

    mode: Literal["agent"] = "agent"
    scope: str
    enabled_tools: tuple[str, ...]
    remaining_model_requests: int
    remaining_tool_calls: int


class GetAgentStatusTool:
    """Expose current mode and remaining deterministic budgets."""

    def __init__(self, *, timeout_seconds: float, max_output_characters: int) -> None:
        self._descriptor = ToolDescriptor(
            name="get_agent_status",
            display_name="查看 Agent 状态",
            description=("查看当前 Agent 模式、对话范围、可用工具和剩余的模型/工具调用预算。"),
            input_model=EmptyToolInput,
            output_model=AgentStatusOutput,
            effect=ToolEffect.READ,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=True,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        del arguments
        return AgentStatusOutput(
            scope=context.identity.conversation.scope,
            enabled_tools=context.enabled_tools,
            remaining_model_requests=max(
                context.limits.max_model_requests - context.model_requests_used,
                0,
            ),
            remaining_tool_calls=max(
                context.limits.max_tool_calls - context.tool_calls_used,
                0,
            ),
        )


class ChannelSettingsOutput(BaseModel):
    """Safe channel policy exposed to the model."""

    scope: str
    chat_enabled: bool
    enabled_agent_tools: tuple[str, ...]


class GetChannelSettingsTool:
    """Read the current channel's feature settings through a project port."""

    def __init__(
        self,
        channels: ChannelToolSettings,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._channels = channels
        self._descriptor = ToolDescriptor(
            name="get_channel_settings",
            display_name="查看频道设置",
            description="查看当前频道启用的聊天和 Agent 工具；私聊返回当前私聊范围。",
            input_model=EmptyToolInput,
            output_model=ChannelSettingsOutput,
            effect=ToolEffect.READ,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=True,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        del arguments
        key = context.identity.conversation
        if key.scope != "channel":
            return ChannelSettingsOutput(
                scope=key.scope,
                chat_enabled=True,
                enabled_agent_tools=context.enabled_tools,
            )
        enabled = await self._channels.enabled_agent_tools(key.area_id, key.channel_id)
        chat_enabled = await self._channels.is_chat_enabled(key.area_id, key.channel_id)
        return ChannelSettingsOutput(
            scope=key.scope,
            chat_enabled=chat_enabled,
            enabled_agent_tools=tuple(sorted(enabled)),
        )


class ReactToMessageInput(BaseModel):
    """A deliberately small allow-list of familiar reactions."""

    model_config = ConfigDict(extra="forbid")

    emoji: Literal["👍", "❤️", "😂", "🎉", "🤔"] = Field(
        description="要添加到用户原消息上的一个表情",
    )


class ReactToMessageOutput(BaseModel):
    """Confirmation returned after the OOPZ side effect."""

    applied: Literal[True] = True
    emoji: str


class ReactToMessageTool:
    """Low-risk, idempotent entertainment write against the source message."""

    def __init__(
        self,
        reactions: ReactionGateway,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._reactions = reactions
        self._descriptor = ToolDescriptor(
            name="react_to_message",
            display_name="回应你的消息",
            description=(
                "给触发当前对话的用户消息添加一个合适的表情反应；仅在确实能增强互动时使用。"
            ),
            input_model=ReactToMessageInput,
            output_model=ReactToMessageOutput,
            effect=ToolEffect.WRITE,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=False,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        validated = ReactToMessageInput.model_validate(arguments)
        await self._reactions.add_reaction(context.identity, validated.emoji)
        return ReactToMessageOutput(emoji=validated.emoji)
