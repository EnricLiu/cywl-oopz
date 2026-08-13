"""Explicit command metadata and collision-safe lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _command_name(value: str, label: str) -> str:
    normalized = value.strip().casefold()
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError(f"{label} must be one non-empty token")
    return normalized


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """User-facing metadata independent from one command implementation."""

    name: str
    summary: str
    category: str
    usage: tuple[str, ...]
    examples: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    hidden: bool = False

    def __post_init__(self) -> None:
        name = _command_name(self.name, "Command name")
        summary = self.summary.strip()
        category = self.category.strip()
        usage = tuple(value.strip() for value in self.usage if value.strip())
        examples = tuple(value.strip() for value in self.examples if value.strip())
        aliases = tuple(_command_name(value, "Command alias") for value in self.aliases)
        if not summary:
            raise ValueError("Command summary must not be empty")
        if not category:
            raise ValueError("Command category must not be empty")
        if not usage:
            raise ValueError("Command usage must not be empty")
        if name in aliases or len(set(aliases)) != len(aliases):
            raise ValueError("Command aliases must be unique and different from its name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "aliases", aliases)

    @classmethod
    def from_command(cls, command: Any) -> CommandSpec:
        """Adapt legacy command attributes into an explicit complete spec."""
        name = str(command.name)
        return cls(
            name=name,
            summary=str(command.description),
            category=str(getattr(command, "category", "其他")),
            usage=tuple(getattr(command, "usage", (name,))),
            examples=tuple(getattr(command, "examples", ())),
            aliases=tuple(getattr(command, "aliases", ())),
            hidden=bool(getattr(command, "hidden", False)),
        )

    def matches(self, name: str) -> bool:
        normalized = name.strip().casefold()
        return normalized == self.name or normalized in self.aliases


class CommandCatalog[EntryT]:
    """Store explicit root entries and resolve canonical names or aliases."""

    def __init__(self) -> None:
        self._entries: dict[str, EntryT] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: CommandSpec, entry: EntryT) -> None:
        occupied = set(self._entries) | set(self._aliases)
        requested = {spec.name, *spec.aliases}
        conflicts = occupied & requested
        if conflicts:
            conflict = sorted(conflicts)[0]
            raise ValueError(f"Command name or alias already registered: {conflict}")
        self._entries[spec.name] = entry
        self._aliases.update({alias: spec.name for alias in spec.aliases})

    def get(self, name: str) -> EntryT | None:
        normalized = name.strip().casefold()
        canonical = self._aliases.get(normalized, normalized)
        return self._entries.get(canonical)

    @property
    def entries(self) -> tuple[EntryT, ...]:
        return tuple(self._entries[name] for name in sorted(self._entries))
