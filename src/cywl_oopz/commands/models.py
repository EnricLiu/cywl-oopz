"""Framework-neutral values for command dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .responses import CommandResponder


def _identifier(value: str, label: str, *, required: bool = True) -> str:
    normalized = value.strip()
    if required and not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) > 256:
        raise ValueError(f"{label} must be at most 256 characters")
    return normalized


class CommandTrigger(StrEnum):
    """Trusted interaction kinds able to invoke one command."""

    TEXT = "text"
    REACTION = "reaction"


class CommandScope(StrEnum):
    """Conversation scopes relevant to command execution."""

    PRIVATE = "private"
    CHANNEL = "channel"


class DispatchStatus(StrEnum):
    """Explicit outcomes used by ingress instead of an ambiguous boolean."""

    NOT_A_COMMAND = "not_a_command"
    UNKNOWN = "unknown"
    IGNORED = "ignored"
    DENIED = "denied"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CommandActor:
    """Stable sender identity projected at the platform boundary."""

    person_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "person_id", _identifier(self.person_id, "Command actor"))


@dataclass(frozen=True, slots=True)
class CommandLocation:
    """Platform-neutral private or area-channel location."""

    scope: CommandScope
    area_id: str = ""
    channel_id: str = ""
    target_person_id: str = ""

    def __post_init__(self) -> None:
        area_id = _identifier(self.area_id, "Area ID", required=False)
        channel_id = _identifier(self.channel_id, "Channel ID", required=False)
        target_person_id = _identifier(
            self.target_person_id,
            "Target person ID",
            required=False,
        )
        if self.scope is CommandScope.CHANNEL and (not area_id or not channel_id):
            raise ValueError("Channel commands require area and channel IDs")
        if self.scope is CommandScope.PRIVATE and area_id:
            raise ValueError("Private commands must not carry an area ID")
        object.__setattr__(self, "area_id", area_id)
        object.__setattr__(self, "channel_id", channel_id)
        object.__setattr__(self, "target_person_id", target_person_id)


@dataclass(frozen=True, slots=True)
class CommandSource:
    """Identifiers from the source event, available for logs and idempotency."""

    message_id: str = ""
    client_message_id: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "message_id",
            _identifier(self.message_id, "Source message ID", required=False),
        )
        object.__setattr__(
            self,
            "client_message_id",
            _identifier(self.client_message_id, "Client message ID", required=False),
        )
        object.__setattr__(
            self,
            "timestamp",
            _identifier(self.timestamp, "Source timestamp", required=False),
        )


@dataclass(frozen=True, slots=True)
class CommandMention:
    """One trusted structured mention attached by OOPZ."""

    person_id: str
    is_bot: bool = False
    bot_type: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "person_id", _identifier(self.person_id, "Mention person"))
        object.__setattr__(
            self,
            "bot_type",
            _identifier(self.bot_type, "Mention bot type", required=False),
        )


@dataclass(frozen=True, slots=True)
class CommandTarget:
    """A referenced or reacted message selected by the invocation."""

    message_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _identifier(self.message_id, "Target message"))


@dataclass(frozen=True, slots=True)
class CommandText:
    """One root command line parsed without interpreting feature arguments."""

    raw: str
    name: str
    raw_tail: str
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        name = self.name.strip().casefold()
        if not name:
            raise ValueError("Command name must not be empty")
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class CommandRequest:
    """One immutable command invocation independent from OOPZ SDK models."""

    trigger: CommandTrigger
    actor: CommandActor
    location: CommandLocation
    source: CommandSource
    responder: CommandResponder
    text: CommandText | None = None
    target: CommandTarget | None = None
    mentions: tuple[CommandMention, ...] = ()

    def __post_init__(self) -> None:
        if self.trigger is CommandTrigger.TEXT and self.text is None:
            raise ValueError("Text command requests require parsed command text")


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    """Observable dispatch result with an explicit ingress-consumption decision."""

    status: DispatchStatus
    command_name: str = ""

    @property
    def consumed(self) -> bool:
        """Return whether later mention/ambient handlers must ignore the event."""
        return self.status is not DispatchStatus.NOT_A_COMMAND
