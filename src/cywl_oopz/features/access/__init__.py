"""Role-based access control for privileged bot capabilities."""

from .models import (
    AccessPrincipal,
    AccessResource,
    AccessResourceKind,
    AccessRole,
    Permission,
    RoleBinding,
    RoleBindingScope,
)
from .policy import RolePermissionPolicy, ScopeMatcher
from .service import AuthorizationService

__all__ = [
    "AccessPrincipal",
    "AccessResource",
    "AccessResourceKind",
    "AccessRole",
    "AuthorizationService",
    "Permission",
    "RoleBinding",
    "RoleBindingScope",
    "RolePermissionPolicy",
    "ScopeMatcher",
]
