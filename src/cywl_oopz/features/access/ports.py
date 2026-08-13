"""Persistence ports for scoped access control."""

from __future__ import annotations

from typing import Protocol

from .models import AccessRole, RoleBinding, RoleBindingScope


class RoleBindingRepository(Protocol):
    """Fresh-read role assignments with idempotent mutation operations."""

    async def list_for_subject(self, subject_person_id: str) -> tuple[RoleBinding, ...]:
        """Load every current binding for one trusted OOPZ person ID."""

    async def list_bindings(
        self,
        *,
        subject_person_id: str | None = None,
    ) -> tuple[RoleBinding, ...]:
        """Load bindings for role-management views in deterministic order."""

    async def grant(self, binding: RoleBinding) -> bool:
        """Insert a binding and report whether it was newly created."""

    async def revoke(
        self,
        subject_person_id: str,
        role: AccessRole,
        scope: RoleBindingScope,
        *,
        area_id: str = "",
        channel_id: str = "",
    ) -> bool:
        """Delete one exact binding and report whether it existed."""
