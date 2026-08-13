"""Deterministic code-owned role and scope policy."""

from __future__ import annotations

from types import MappingProxyType

from .models import (
    AccessResource,
    AccessResourceKind,
    AccessRole,
    Permission,
    RoleBinding,
    RoleBindingScope,
)


class RolePermissionPolicy:
    """Map the fixed role set to concrete privileged operations."""

    _ROLE_PERMISSIONS = MappingProxyType(
        {
            AccessRole.OWNER: frozenset(Permission),
            AccessRole.ADMIN: frozenset(
                {
                    Permission.BOT_REBOOT,
                    Permission.AGENT_RESPONSE_DEBUG,
                    Permission.BOT_MESSAGE_RECALL,
                    Permission.CHANNEL_INITIALIZE,
                    Permission.RBAC_VIEW,
                }
            ),
            AccessRole.MODERATOR: frozenset({Permission.BOT_MESSAGE_RECALL}),
        }
    )

    @classmethod
    def permissions(cls, role: AccessRole) -> frozenset[Permission]:
        return cls._ROLE_PERMISSIONS[role]

    @classmethod
    def allows(cls, role: AccessRole, permission: Permission) -> bool:
        return permission in cls.permissions(role)


class ScopeMatcher:
    """Match one persisted binding against one runtime resource."""

    @staticmethod
    def matches(binding: RoleBinding, resource: AccessResource) -> bool:
        if binding.scope is RoleBindingScope.GLOBAL:
            return True
        if resource.kind in {AccessResourceKind.GLOBAL, AccessResourceKind.PRIVATE}:
            return False
        if binding.scope is RoleBindingScope.AREA:
            return binding.area_id == resource.area_id
        return (
            resource.kind is AccessResourceKind.CHANNEL
            and binding.area_id == resource.area_id
            and binding.channel_id == resource.channel_id
        )
