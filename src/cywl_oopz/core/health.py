"""Small in-process health registry without sensitive diagnostic output."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class HealthState(StrEnum):
    """The public state of one application component."""

    PENDING = "pending"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """A sanitised point-in-time view of one component."""

    name: str
    state: HealthState
    checked_at: datetime
    detail: str = ""


class HealthRegistry:
    """Stores component status for commands and logs in the current process."""

    def __init__(self) -> None:
        self._checks: dict[str, HealthCheck] = {}

    def mark(self, name: str, state: HealthState, detail: str = "") -> None:
        """Record a status using an intentionally short, non-sensitive detail."""
        self._checks[name] = HealthCheck(
            name=name,
            state=state,
            checked_at=datetime.now(UTC),
            detail=detail,
        )

    def snapshot(self) -> tuple[HealthCheck, ...]:
        """Return checks in stable order for a user-facing status response."""
        return tuple(self._checks[name] for name in sorted(self._checks))
