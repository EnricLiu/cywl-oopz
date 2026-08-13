"""Typed command definition contracts and common parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from cywl_oopz.features.access.models import AccessResource, Permission

from .catalog import CommandSpec
from .models import CommandRequest


@dataclass(frozen=True, slots=True)
class AccessRequirement:
    """One concrete permission check resolved from typed command arguments."""

    permission: Permission
    resource: AccessResource


class ExecutionMode(StrEnum):
    """Whether dispatch waits inline or hands work to the application supervisor."""

    INLINE = "inline"
    BACKGROUND = "background"


@dataclass(frozen=True, slots=True)
class CommandExecutionPolicy:
    """Lifecycle policy declared beside a typed command definition."""

    mode: ExecutionMode = ExecutionMode.INLINE
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("Command timeout must be positive")


class CommandUsageError(ValueError):
    """Expected syntax/context rejection rendered without an exception traceback."""

    def __init__(
        self,
        message: str,
        *,
        include_usage: bool = True,
    ) -> None:
        self.user_message = message.strip()
        self.include_usage = include_usage
        super().__init__(self.user_message)

    def render(self, spec: CommandSpec, prefix: str) -> str:
        lines = [self.user_message] if self.user_message else []
        if self.include_usage:
            lines.append("用法：")
            lines.extend(f"{prefix}{usage}" for usage in spec.usage)
        return "\n".join(lines)


class ArgumentParser[ArgsT](Protocol):
    """Parse one request into immutable feature arguments without doing I/O."""

    def parse(self, request: CommandRequest) -> ArgsT:
        """Return typed arguments or raise ``CommandUsageError``."""


class CommandHandler[ArgsT](Protocol):
    """Execute one application use case using project-owned request values."""

    async def handle(self, request: CommandRequest, arguments: ArgsT) -> None:
        """Run after parsing and authorization complete."""


class CommandAuthorization[ArgsT](Protocol):
    """Resolve availability and requirements from the same typed arguments."""

    def is_available(self, request: CommandRequest) -> bool:
        """Return whether any path is meaningful in the current context."""

    def requirement(
        self,
        request: CommandRequest,
        arguments: ArgsT,
    ) -> AccessRequirement | None:
        """Return the exact execution requirement, or none for a public path."""

    def visibility_requirement(
        self,
        request: CommandRequest,
    ) -> AccessRequirement | None:
        """Return the coarse requirement used by command discovery."""


class PublicCommandAuthorization[ArgsT]:
    """Allow a typed command in every supported request context."""

    def is_available(self, request: CommandRequest) -> bool:
        del request
        return True

    def requirement(
        self,
        request: CommandRequest,
        arguments: ArgsT,
    ) -> None:
        del request, arguments
        return None

    def visibility_requirement(self, request: CommandRequest) -> None:
        del request
        return None


@dataclass(frozen=True, slots=True)
class CommandDefinition[ArgsT]:
    """Complete typed registration assembled explicitly by the composition root."""

    spec: CommandSpec
    parser: ArgumentParser[ArgsT]
    handler: CommandHandler[ArgsT]
    authorization: CommandAuthorization[ArgsT] = field(default_factory=PublicCommandAuthorization)
    execution: CommandExecutionPolicy = field(default_factory=CommandExecutionPolicy)


@dataclass(frozen=True, slots=True)
class NoArguments:
    """Typed marker proving a command received no arguments."""


class NoArgumentsParser:
    """Reject accidental arguments before a state-changing handler runs."""

    def parse(self, request: CommandRequest) -> NoArguments:
        text = request.text
        if text is None or text.tokens:
            raise CommandUsageError("此命令不接受额外参数。")
        return NoArguments()
