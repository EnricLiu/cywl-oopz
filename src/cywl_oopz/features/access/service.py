"""Fresh-read authorization service shared by commands and Agent tools."""

from __future__ import annotations

from cywl_oopz.core.errors import AuthorizationError

from .models import AccessPrincipal, AccessResource, AccessRole, Permission, RoleBinding
from .policy import RolePermissionPolicy, ScopeMatcher
from .ports import RoleBindingRepository


class AuthorizationService:
    """Authorize each privileged operation from current PostgreSQL bindings."""

    def __init__(
        self,
        repository: RoleBindingRepository,
        bootstrap_owner_ids: frozenset[str] = frozenset(),
    ) -> None:
        normalized = frozenset(value.strip() for value in bootstrap_owner_ids if value.strip())
        if any(len(value) > 128 for value in normalized):
            raise ValueError("Bootstrap owner IDs must be at most 128 characters")
        self._repository = repository
        self._bootstrap_owner_ids = normalized

    def is_bootstrap_owner(self, principal: AccessPrincipal) -> bool:
        return principal.person_id in self._bootstrap_owner_ids

    async def bindings(self, principal: AccessPrincipal) -> tuple[RoleBinding, ...]:
        """Expose current bindings for role-management views without caching."""
        return await self._repository.list_for_subject(principal.person_id)

    async def allows(
        self,
        principal: AccessPrincipal,
        permission: Permission,
        resource: AccessResource,
    ) -> bool:
        """Return a fresh authorization decision."""
        if self.is_bootstrap_owner(principal):
            return True
        bindings = await self.bindings(principal)
        return any(
            RolePermissionPolicy.allows(binding.role, permission)
            and ScopeMatcher.matches(binding, resource)
            for binding in bindings
        )

    async def require(
        self,
        principal: AccessPrincipal,
        permission: Permission,
        resource: AccessResource,
    ) -> None:
        """Raise the project's expected authorization error when denied."""
        if not await self.allows(principal, permission, resource):
            raise AuthorizationError("Permission denied")

    async def effective_roles(
        self,
        principal: AccessPrincipal,
        resource: AccessResource,
    ) -> frozenset[AccessRole]:
        """Return current matching roles for `/role me` and diagnostics."""
        roles = {
            binding.role
            for binding in await self.bindings(principal)
            if ScopeMatcher.matches(binding, resource)
        }
        if self.is_bootstrap_owner(principal):
            roles.add(AccessRole.OWNER)
        return frozenset(roles)
