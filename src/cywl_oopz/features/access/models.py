"""Framework-neutral values for scoped role-based access control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AccessRole(StrEnum):
    """Small fixed role set for the single-bot deployment."""

    OWNER = "owner"
    ADMIN = "admin"
    MODERATOR = "moderator"


class Permission(StrEnum):
    """Stable privileged operations checked at application boundaries."""

    BOT_REBOOT = "bot.reboot"
    AGENT_RESPONSE_DEBUG = "agent_response.debug"
    BOT_MESSAGE_RECALL = "bot_message.recall"
    CHANNEL_INITIALIZE = "channel.initialize"
    RBAC_VIEW = "rbac.view"
    RBAC_MANAGE = "rbac.manage"


class RoleBindingScope(StrEnum):
    """Scopes that can be persisted for one role binding."""

    GLOBAL = "global"
    AREA = "area"
    CHANNEL = "channel"


class AccessResourceKind(StrEnum):
    """Runtime resource kinds against which bindings are evaluated."""

    GLOBAL = "global"
    PRIVATE = "private"
    AREA = "area"
    CHANNEL = "channel"


def _identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(normalized) > 128:
        raise ValueError(f"{label} must be at most 128 characters")
    return normalized


@dataclass(frozen=True, slots=True)
class AccessPrincipal:
    """Trusted OOPZ message sender identity."""

    person_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "person_id", _identifier(self.person_id, "Person ID"))


@dataclass(frozen=True, slots=True)
class AccessResource:
    """One normalized authorization target independent from OOPZ SDK models."""

    kind: AccessResourceKind
    area_id: str = ""
    channel_id: str = ""

    def __post_init__(self) -> None:
        area_id = self.area_id.strip()
        channel_id = self.channel_id.strip()
        if len(area_id) > 128 or len(channel_id) > 128:
            raise ValueError("Resource identifiers must be at most 128 characters")
        if self.kind in {AccessResourceKind.GLOBAL, AccessResourceKind.PRIVATE}:
            if area_id or channel_id:
                raise ValueError(f"{self.kind.value} resources must not carry an address")
        elif self.kind is AccessResourceKind.AREA:
            if not area_id or channel_id:
                raise ValueError("Area resources require only an area ID")
        elif self.kind is AccessResourceKind.CHANNEL:
            if not area_id or not channel_id:
                raise ValueError("Channel resources require area and channel IDs")
        else:  # pragma: no cover - StrEnum construction prevents this branch
            raise ValueError("Unknown access resource kind")
        object.__setattr__(self, "area_id", area_id)
        object.__setattr__(self, "channel_id", channel_id)

    @classmethod
    def global_resource(cls) -> AccessResource:
        return cls(AccessResourceKind.GLOBAL)

    @classmethod
    def private(cls) -> AccessResource:
        return cls(AccessResourceKind.PRIVATE)

    @classmethod
    def area(cls, area_id: str) -> AccessResource:
        return cls(AccessResourceKind.AREA, area_id=area_id)

    @classmethod
    def channel(cls, area_id: str, channel_id: str) -> AccessResource:
        return cls(AccessResourceKind.CHANNEL, area_id=area_id, channel_id=channel_id)


@dataclass(frozen=True, slots=True)
class RoleBinding:
    """One persisted role assignment and its exact resource address."""

    subject_person_id: str
    role: AccessRole
    scope: RoleBindingScope
    area_id: str = ""
    channel_id: str = ""
    granted_by_person_id: str = ""

    def __post_init__(self) -> None:
        subject = _identifier(self.subject_person_id, "Role subject")
        area_id = self.area_id.strip()
        channel_id = self.channel_id.strip()
        granted_by = self.granted_by_person_id.strip()
        if any(len(value) > 128 for value in (area_id, channel_id, granted_by)):
            raise ValueError("Role binding identifiers must be at most 128 characters")
        if self.scope is RoleBindingScope.GLOBAL:
            if area_id or channel_id:
                raise ValueError("Global role bindings must not carry an address")
        elif self.scope is RoleBindingScope.AREA:
            if not area_id or channel_id:
                raise ValueError("Area role bindings require only an area ID")
        elif self.scope is RoleBindingScope.CHANNEL:
            if not area_id or not channel_id:
                raise ValueError("Channel role bindings require area and channel IDs")
        if self.role is AccessRole.OWNER and self.scope is not RoleBindingScope.GLOBAL:
            raise ValueError("Owner role bindings must be global")
        object.__setattr__(self, "subject_person_id", subject)
        object.__setattr__(self, "area_id", area_id)
        object.__setattr__(self, "channel_id", channel_id)
        object.__setattr__(self, "granted_by_person_id", granted_by)
