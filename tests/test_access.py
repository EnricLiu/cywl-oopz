from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cywl_oopz.core.errors import AuthorizationError
from cywl_oopz.features.access.models import (
    AccessPrincipal,
    AccessResource,
    AccessRole,
    Permission,
    RoleBinding,
    RoleBindingScope,
)
from cywl_oopz.features.access.policy import RolePermissionPolicy, ScopeMatcher
from cywl_oopz.features.access.service import AuthorizationService


@dataclass
class InMemoryRoleBindings:
    records: list[RoleBinding] = field(default_factory=list)
    reads: int = 0

    async def list_for_subject(self, subject_person_id: str) -> tuple[RoleBinding, ...]:
        self.reads += 1
        return tuple(
            binding for binding in self.records if binding.subject_person_id == subject_person_id
        )


def binding(
    role: AccessRole,
    scope: RoleBindingScope,
    *,
    area_id: str = "",
    channel_id: str = "",
) -> RoleBinding:
    return RoleBinding(
        subject_person_id="person",
        role=role,
        scope=scope,
        area_id=area_id,
        channel_id=channel_id,
    )


def test_access_values_reject_invalid_addresses() -> None:
    assert AccessPrincipal(" person ").person_id == "person"
    assert AccessResource.channel(" area ", " channel ") == AccessResource.channel(
        "area", "channel"
    )

    with pytest.raises(ValueError, match="Area resources"):
        AccessResource.area("")
    with pytest.raises(ValueError, match="Channel resources"):
        AccessResource.channel("area", "")
    with pytest.raises(ValueError, match="Owner role bindings"):
        binding(AccessRole.OWNER, RoleBindingScope.AREA, area_id="area")
    with pytest.raises(ValueError, match="Global role bindings"):
        binding(AccessRole.ADMIN, RoleBindingScope.GLOBAL, area_id="area")


def test_role_permission_matrix_is_explicit() -> None:
    assert RolePermissionPolicy.permissions(AccessRole.OWNER) == frozenset(Permission)
    assert RolePermissionPolicy.permissions(AccessRole.ADMIN) == frozenset(
        {
            Permission.BOT_REBOOT,
            Permission.AGENT_RESPONSE_DEBUG,
            Permission.BOT_MESSAGE_RECALL,
            Permission.CHANNEL_INITIALIZE,
            Permission.RBAC_VIEW,
        }
    )
    assert RolePermissionPolicy.permissions(AccessRole.MODERATOR) == frozenset(
        {Permission.BOT_MESSAGE_RECALL}
    )


@pytest.mark.parametrize(
    ("role_binding", "resource", "expected"),
    [
        (
            binding(AccessRole.ADMIN, RoleBindingScope.GLOBAL),
            AccessResource.global_resource(),
            True,
        ),
        (
            binding(AccessRole.ADMIN, RoleBindingScope.GLOBAL),
            AccessResource.private(),
            True,
        ),
        (
            binding(AccessRole.ADMIN, RoleBindingScope.AREA, area_id="area"),
            AccessResource.area("area"),
            True,
        ),
        (
            binding(AccessRole.ADMIN, RoleBindingScope.AREA, area_id="area"),
            AccessResource.channel("area", "channel"),
            True,
        ),
        (
            binding(AccessRole.ADMIN, RoleBindingScope.AREA, area_id="area"),
            AccessResource.channel("other", "channel"),
            False,
        ),
        (
            binding(
                AccessRole.ADMIN,
                RoleBindingScope.CHANNEL,
                area_id="area",
                channel_id="channel",
            ),
            AccessResource.channel("area", "channel"),
            True,
        ),
        (
            binding(
                AccessRole.ADMIN,
                RoleBindingScope.CHANNEL,
                area_id="area",
                channel_id="channel",
            ),
            AccessResource.area("area"),
            False,
        ),
        (
            binding(AccessRole.ADMIN, RoleBindingScope.AREA, area_id="area"),
            AccessResource.private(),
            False,
        ),
        (
            binding(AccessRole.ADMIN, RoleBindingScope.AREA, area_id="area"),
            AccessResource.global_resource(),
            False,
        ),
    ],
)
def test_scope_match_matrix(
    role_binding: RoleBinding,
    resource: AccessResource,
    expected: bool,
) -> None:
    assert ScopeMatcher.matches(role_binding, resource) is expected


@pytest.mark.asyncio
async def test_authorization_reads_current_bindings_for_every_operation() -> None:
    repository = InMemoryRoleBindings()
    service = AuthorizationService(repository)
    principal = AccessPrincipal("person")
    resource = AccessResource.channel("area", "channel")

    assert not await service.allows(principal, Permission.BOT_MESSAGE_RECALL, resource)
    repository.records.append(binding(AccessRole.MODERATOR, RoleBindingScope.AREA, area_id="area"))
    assert await service.allows(principal, Permission.BOT_MESSAGE_RECALL, resource)
    repository.records.clear()
    assert not await service.allows(principal, Permission.BOT_MESSAGE_RECALL, resource)
    assert repository.reads == 3


@pytest.mark.asyncio
async def test_scope_prevents_area_admin_from_global_reboot() -> None:
    service = AuthorizationService(
        InMemoryRoleBindings([binding(AccessRole.ADMIN, RoleBindingScope.AREA, area_id="area")])
    )
    principal = AccessPrincipal("person")

    assert await service.allows(
        principal,
        Permission.CHANNEL_INITIALIZE,
        AccessResource.channel("area", "channel"),
    )
    with pytest.raises(AuthorizationError):
        await service.require(
            principal,
            Permission.BOT_REBOOT,
            AccessResource.global_resource(),
        )


@pytest.mark.asyncio
async def test_bootstrap_owner_is_global_without_authorization_repository_read() -> None:
    repository = InMemoryRoleBindings()
    service = AuthorizationService(repository, frozenset({"owner"}))
    principal = AccessPrincipal("owner")

    assert await service.allows(
        principal,
        Permission.RBAC_MANAGE,
        AccessResource.global_resource(),
    )
    assert repository.reads == 0
    assert await service.effective_roles(principal, AccessResource.private()) == frozenset(
        {AccessRole.OWNER}
    )
    # `/role me` also reads database bindings so it can show both assignment sources.
    assert repository.reads == 1
