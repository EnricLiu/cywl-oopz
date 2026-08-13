"""Deterministic tool visibility and authorization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.access.models import Permission
from cywl_oopz.features.agent.models import AgentIdentity, AgentModelRef, ModelCapability

from .models import ToolDescriptor, ToolEffect, ToolExecutionContext
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


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


class AgentToolAuthorization(Protocol):
    """Fresh scoped RBAC decision for one trusted Agent identity."""

    async def allows(self, identity: AgentIdentity, permission: Permission) -> bool:
        """Return whether the exact permission is currently granted."""


@dataclass(frozen=True, slots=True)
class AvailableTool:
    """Safe metadata suitable for commands and engine requests."""

    name: str
    description: str
    effect: ToolEffect


class ToolPolicy:
    """Authorize execution from server facts, never model arguments."""

    def __init__(self, authorization: AgentToolAuthorization | None = None) -> None:
        self._authorization = authorization

    async def denial_reason(
        self,
        context: ToolExecutionContext,
        descriptor: ToolDescriptor,
    ) -> str:
        """Return a stable denial code, rechecking RBAC at execution time."""
        if descriptor.name not in context.enabled_tools:
            return "tool_not_enabled"
        return await self.permission_denial(context.identity, descriptor)

    async def permission_denial(
        self,
        identity: AgentIdentity,
        descriptor: ToolDescriptor,
    ) -> str:
        permission = descriptor.required_permission
        if permission is None:
            return ""
        if self._authorization is None:
            return "permission_denied"
        try:
            allowed = await self._authorization.allows(identity, permission)
        except DatabaseError:
            logger.warning(
                "Agent tool authorization unavailable: principal=%s permission=%s tool=%s",
                opaque_ref(identity.person_id),
                permission.value,
                descriptor.name,
            )
            return "authorization_unavailable"
        return "" if allowed else "permission_denied"


class ToolAvailabilityService:
    """Intersect registry, application, channel, mode, and model capability."""

    def __init__(
        self,
        registry: ToolRegistry,
        channels: ChannelToolSettings,
        application_enabled_tools: tuple[str, ...],
        policy: ToolPolicy | None = None,
    ) -> None:
        unknown = frozenset(application_enabled_tools).difference(registry.names)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown configured Agent tools: {names}")
        self._registry = registry
        self._channels = channels
        self._application_enabled = frozenset(application_enabled_tools)
        self._policy = policy or ToolPolicy()

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
        available: list[AvailableTool] = []
        for item in self._registry.descriptors(tuple(enabled)):
            if await self._policy.permission_denial(identity, item):
                continue
            available.append(AvailableTool(item.name, item.description, item.effect))
        return tuple(available)

    async def names(
        self,
        identity: AgentIdentity,
        model: AgentModelRef,
    ) -> tuple[str, ...]:
        """Return names for an already trusted identity."""
        available = await self.resolve(identity, model)
        return tuple(item.name for item in available)
