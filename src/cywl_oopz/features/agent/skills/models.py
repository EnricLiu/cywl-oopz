"""Immutable domain values for database-backed Agent skills."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
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


class SkillOwnershipKind(StrEnum):
    """Who controls one persisted Skill definition."""

    BUILTIN = "builtin"
    PERSONAL = "personal"


class SkillAccessKind(StrEnum):
    """How one caller may access a Skill in their library."""

    BUILTIN = "builtin"
    OWNED = "owned"
    SHARED = "shared"


class SkillShareStatus(StrEnum):
    """Recipient-controlled lifecycle of one Skill invitation."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"


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
    ownership_kind: SkillOwnershipKind = SkillOwnershipKind.BUILTIN
    owner_person_id: str | None = None
    enabled: bool = True
    archived_at: datetime | None = None

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
        owner_person_id = self.owner_person_id.strip() if self.owner_person_id is not None else None
        if self.ownership_kind is SkillOwnershipKind.BUILTIN:
            if owner_person_id is not None:
                raise ValueError("Builtin Skill must not have an owner")
            if self.archived_at is not None:
                raise ValueError("Builtin Skill must not be user-archived")
        elif not owner_person_id:
            raise ValueError("Personal Skill requires an owner")
        if self.archived_at is not None and self.enabled:
            raise ValueError("Archived Skill must not be enabled")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "required_tools", required_tools)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "owner_person_id", owner_person_id)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {str(key): _freeze_json(value) for key, value in self.metadata.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class AgentSkillDiscovery:
    """Metadata-only projection visible to one caller during discovery."""

    id: UUID
    name: str
    display_name: str
    description: str
    version: str
    revision: int
    required_tools: frozenset[str]
    access: SkillAccessKind

    def __post_init__(self) -> None:
        name = self.name.strip()
        display_name = self.display_name.strip()
        description = self.description.strip()
        version = self.version.strip()
        if not _SKILL_NAME.fullmatch(name):
            raise ValueError("Skill name must be a stable lowercase kebab-case name")
        _validate_line(display_name, "Skill display name", 80)
        _validate_text(description, "Skill description", 1_024)
        _validate_line(version, "Skill version", 64)
        if self.revision <= 0:
            raise ValueError("Skill revision must be positive")
        required_tools = frozenset(tool.strip() for tool in self.required_tools)
        if any(not _TOOL_NAME.fullmatch(tool) for tool in required_tools):
            raise ValueError("Skill required tools must use stable tool names")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "required_tools", required_tools)


@dataclass(frozen=True, slots=True)
class AgentSkillShare:
    """One invitation or accepted read grant for a personal Skill."""

    id: UUID
    skill_id: UUID
    recipient_person_id: str
    status: SkillShareStatus
    created_at: datetime
    updated_at: datetime
    responded_at: datetime | None = None

    def __post_init__(self) -> None:
        recipient = self.recipient_person_id.strip()
        if not recipient:
            raise ValueError("Skill share recipient must not be empty")
        if self.status is SkillShareStatus.PENDING and self.responded_at is not None:
            raise ValueError("Pending Skill share must not have a response time")
        if self.status is not SkillShareStatus.PENDING and self.responded_at is None:
            raise ValueError("Responded Skill share requires a response time")
        object.__setattr__(self, "recipient_person_id", recipient)


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
