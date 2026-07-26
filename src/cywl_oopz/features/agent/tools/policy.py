"""Deterministic tool visibility and authorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cywl_oopz.features.agent.models import AgentIdentity, AgentModelRef, ModelCapability

from .models import ToolDescriptor, ToolEffect, ToolExecutionContext
from .registry import ToolRegistry


class ChannelToolSettings(Protocol):
    """Read boundary for per-channel Agent tool enablement."""

    async def is_chat_enabled(self, area_id: str, channel_id: str) -> bool:
        """Return whether ambient chat is enabled."""

    async def enabled_agent_tools(
        self,
        area_id: str,
        channel_id: str,
    ) -> frozenset[str]:
        """Return stable tool names enabled for one channel."""


@dataclass(frozen=True, slots=True)
class AvailableTool:
    """Safe metadata suitable for commands and engine requests."""

    name: str
    description: str
    effect: ToolEffect


class ToolPolicy:
    """Authorize execution from server facts, never model arguments."""

    @staticmethod
    def allows(context: ToolExecutionContext, descriptor: ToolDescriptor) -> bool:
        """Return whether this exact call is authorized."""
        if descriptor.name not in context.enabled_tools:
            return False
        if descriptor.effect is ToolEffect.ADMIN and not context.identity.is_administrator:
            return False
        return True


class ToolAvailabilityService:
    """Intersect registry, application, channel, mode, and model capability."""

    def __init__(
        self,
        registry: ToolRegistry,
        channels: ChannelToolSettings,
        application_enabled_tools: tuple[str, ...],
    ) -> None:
        unknown = frozenset(application_enabled_tools).difference(registry.names)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown configured Agent tools: {names}")
        self._registry = registry
        self._channels = channels
        self._application_enabled = frozenset(application_enabled_tools)

    async def resolve(
        self,
        identity: AgentIdentity,
        model: AgentModelRef,
    ) -> tuple[AvailableTool, ...]:
        """Return the exact safe tool set visible for one run."""
        if ModelCapability.TOOL_CALLING not in model.capabilities:
            return ()
        key = identity.conversation
        enabled = self._application_enabled
        if key.scope == "channel":
            channel_enabled = await self._channels.enabled_agent_tools(
                key.area_id,
                key.channel_id,
            )
            enabled = enabled.intersection(channel_enabled)
        return tuple(
            AvailableTool(item.name, item.description, item.effect)
            for item in self._registry.descriptors(tuple(enabled))
            if item.effect is not ToolEffect.ADMIN or identity.is_administrator
        )

    async def names(
        self,
        identity: AgentIdentity,
        model: AgentModelRef,
    ) -> tuple[str, ...]:
        """Return names for an already trusted identity."""
        available = await self.resolve(identity, model)
        return tuple(item.name for item in available)
