"""OOPZ event-context projection into trusted access-control values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cywl_oopz.features.access.models import AccessPrincipal, AccessResource


@dataclass(frozen=True, slots=True)
class OopzAccessInvocation:
    """Trusted principal and current resource extracted at the SDK boundary."""

    principal: AccessPrincipal
    resource: AccessResource

    @classmethod
    def from_context(cls, context: Any) -> OopzAccessInvocation:
        event = getattr(context, "event", None)
        message = getattr(event, "message", None)
        if message is None:
            raise ValueError("Command authorization requires an OOPZ message event")
        principal = AccessPrincipal(str(getattr(message, "sender_id", "")))
        if bool(getattr(event, "is_private", False)):
            return cls(principal, AccessResource.private())
        return cls(
            principal,
            AccessResource.channel(
                str(getattr(message, "area", "")),
                str(getattr(message, "channel", "")),
            ),
        )
