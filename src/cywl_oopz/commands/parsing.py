"""Small project-owned lexer for root text commands."""

from __future__ import annotations

from .models import CommandText


class CommandTextParser:
    """Recognize a configured prefix while preserving the untouched argument tail."""

    def __init__(self, prefix: str) -> None:
        if not prefix:
            raise ValueError("Command prefix must not be empty")
        self.prefix = prefix

    def parse(self, text: str) -> CommandText | None:
        """Parse only the root name; feature parsers own all remaining syntax."""
        content = text.strip()
        if not content.startswith(self.prefix):
            return None

        command_line = content[len(self.prefix) :].lstrip()
        parts = command_line.split(maxsplit=1)
        if not parts:
            return None
        raw_tail = parts[1] if len(parts) == 2 else ""
        return CommandText(
            raw=content,
            name=parts[0],
            raw_tail=raw_tail,
            tokens=tuple(raw_tail.split()),
        )
