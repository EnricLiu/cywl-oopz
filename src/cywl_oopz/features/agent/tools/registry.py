"""Immutable registry for application-constructed Agent tools."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from .models import ToolDescriptor
from .ports import AgentTool


class ToolRegistry:
    """Resolve only explicit tool objects; dynamic imports are never accepted."""

    def __init__(self, tools: Iterable[AgentTool]) -> None:
        registered: dict[str, AgentTool] = {}
        for tool in tools:
            name = tool.descriptor.name
            if name in registered:
                raise ValueError(f"Duplicate Agent tool name: {name}")
            registered[name] = tool
        self._tools = MappingProxyType(registered)

    def get(self, name: str) -> AgentTool | None:
        """Return one registered tool by its exact stable name."""
        return self._tools.get(name)

    def descriptors(self, names: tuple[str, ...] | None = None) -> tuple[ToolDescriptor, ...]:
        """Return descriptors in stable name order, optionally restricted by name."""
        allowed = None if names is None else frozenset(names)
        return tuple(
            self._tools[name].descriptor
            for name in sorted(self._tools)
            if allowed is None or name in allowed
        )

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in stable order."""
        return tuple(sorted(self._tools))
