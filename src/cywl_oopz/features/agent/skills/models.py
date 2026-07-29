"""Immutable domain values for database-backed Agent skills."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from uuid import UUID

_SKILL_NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_RESOURCE_KEY = re.compile(r"^[a-z][a-z0-9-]{0,159}$")
_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_TEXT_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "text/markdown",
        "text/plain",
    }
)


class SkillResourceKind(StrEnum):
    """Supported non-executable resource categories."""

    REFERENCE = "reference"
    TEMPLATE = "template"
    EXAMPLE = "example"


@dataclass(frozen=True, slots=True)
class AgentSkillResource:
    """One bounded text resource belonging to an Agent skill."""

    id: UUID
    key: str
    display_name: str
    description: str
    kind: SkillResourceKind
    media_type: str
    content: str
    position: int

    def __post_init__(self) -> None:
        key = self.key.strip()
        display_name = self.display_name.strip()
        description = self.description.strip()
        media_type = self.media_type.strip().casefold()
        if not _RESOURCE_KEY.fullmatch(key):
            raise ValueError("Skill resource key must be a stable lowercase kebab-case name")
        _validate_line(display_name, "Skill resource display name", 120)
        _validate_text(description, "Skill resource description", 500)
        if media_type not in _TEXT_MEDIA_TYPES:
            raise ValueError("Skill resource media type must be supported text")
        _validate_text(self.content, "Skill resource content", 20_000)
        if self.position <= 0:
            raise ValueError("Skill resource position must be positive")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "media_type", media_type)


@dataclass(frozen=True, slots=True)
class AgentSkill:
    """One validated skill bundle pinned into an Agent run snapshot."""

    id: UUID
    name: str
    display_name: str
    description: str
    instructions: str
    version: str
    revision: int
    required_tools: frozenset[str]
    resources: tuple[AgentSkillResource, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        name = self.name.strip()
        display_name = self.display_name.strip()
        description = self.description.strip()
        instructions = self.instructions.strip()
        version = self.version.strip()
        if not _SKILL_NAME.fullmatch(name):
            raise ValueError("Skill name must be a stable lowercase kebab-case name")
        _validate_line(display_name, "Skill display name", 80)
        _validate_text(description, "Skill description", 1_024)
        _validate_text(instructions, "Skill instructions", 20_000)
        _validate_line(version, "Skill version", 64)
        if self.revision <= 0:
            raise ValueError("Skill revision must be positive")
        if any(not isinstance(tool, str) for tool in self.required_tools):
            raise ValueError("Skill required tools must be strings")
        required_tools = frozenset(tool.strip() for tool in self.required_tools)
        if any(not _TOOL_NAME.fullmatch(tool) for tool in required_tools):
            raise ValueError("Skill required tools must use stable tool names")
        resources = tuple(sorted(self.resources, key=lambda item: item.position))
        if len({item.id for item in resources}) != len(resources):
            raise ValueError("Skill resources contain duplicate IDs")
        if len({item.key for item in resources}) != len(resources):
            raise ValueError("Skill resources contain duplicate keys")
        if len({item.position for item in resources}) != len(resources):
            raise ValueError("Skill resources contain duplicate positions")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Skill metadata must be a mapping")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "required_tools", required_tools)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {str(key): _freeze_json(value) for key, value in self.metadata.items()}
            ),
        )


def _validate_line(value: str, label: str, limit: int) -> None:
    if not value or len(value) > limit or "\n" in value or "\r" in value:
        raise ValueError(f"{label} must be one non-empty line of at most {limit} characters")


def _validate_text(value: str, label: str, limit: int) -> None:
    if not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must contain at most {limit} non-empty characters")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value
