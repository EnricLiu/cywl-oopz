"""Role-binding management use cases with bounded visibility rules."""

from __future__ import annotations

from .models import (
    AccessPrincipal,
    AccessResource,
    AccessResourceKind,
    AccessRole,
    Permission,
    RoleBinding,
    RoleBindingScope,
)
from .ports import RoleBindingRepository
from .service import AuthorizationService


class RoleAdministrationService:
    """List and mutate role bindings without depending on OOPZ contexts."""

    def __init__(
        self,
        repository: RoleBindingRepository,
        authorizer: AuthorizationService,
    ) -> None:
        self._repository = repository
        self._authorizer = authorizer

    async def grant(
        self,
        actor: AccessPrincipal,
        subject: AccessPrincipal,
        role: AccessRole,
        scope: RoleBindingScope,
        current_resource: AccessResource,
    ) -> bool:
        area_id, channel_id = self._binding_address(scope, current_resource)
        return await self._repository.grant(
            RoleBinding(
                subject_person_id=subject.person_id,
                role=role,
                scope=scope,
                area_id=area_id,
                channel_id=channel_id,
                granted_by_person_id=actor.person_id,
            )
        )

    async def revoke(
        self,
        subject: AccessPrincipal,
        role: AccessRole,
        scope: RoleBindingScope,
        current_resource: AccessResource,
    ) -> bool:
        if self._authorizer.is_bootstrap_owner(subject):
            raise ValueError("Bootstrap owner roles cannot be revoked")
        area_id, channel_id = self._binding_address(scope, current_resource)
        return await self._repository.revoke(
            subject.person_id,
            role,
            scope,
            area_id=area_id,
            channel_id=channel_id,
        )

    async def visible_bindings(
        self,
        viewer: AccessPrincipal,
        current_resource: AccessResource,
        *,
        subject: AccessPrincipal | None = None,
    ) -> tuple[RoleBinding, ...]:
        """Return only assignments visible within the viewer's effective scope."""
        records = await self._repository.list_bindings(
            subject_person_id=subject.person_id if subject is not None else None
        )
        global_resource = AccessResource.global_resource()
        if await self._authorizer.allows(viewer, Permission.RBAC_MANAGE, global_resource):
            return records
        if await self._authorizer.allows(viewer, Permission.RBAC_VIEW, global_resource):
            return tuple(record for record in records if record.role is not AccessRole.OWNER)
        if current_resource.kind not in {
            AccessResourceKind.AREA,
            AccessResourceKind.CHANNEL,
        }:
            return ()
        area_resource = AccessResource.area(current_resource.area_id)
        if await self._authorizer.allows(viewer, Permission.RBAC_VIEW, area_resource):
            return tuple(
                record
                for record in records
                if record.scope is not RoleBindingScope.GLOBAL
                and record.area_id == current_resource.area_id
            )
        if current_resource.kind is AccessResourceKind.CHANNEL and await self._authorizer.allows(
            viewer,
            Permission.RBAC_VIEW,
            current_resource,
        ):
            return tuple(
                record
                for record in records
                if record.scope is RoleBindingScope.CHANNEL
                and record.area_id == current_resource.area_id
                and record.channel_id == current_resource.channel_id
            )
        return ()

    @staticmethod
    def resource_for_scope(
        scope: RoleBindingScope,
        current_resource: AccessResource,
    ) -> AccessResource:
        area_id, channel_id = RoleAdministrationService._binding_address(scope, current_resource)
        if scope is RoleBindingScope.GLOBAL:
            return AccessResource.global_resource()
        if scope is RoleBindingScope.AREA:
            return AccessResource.area(area_id)
        return AccessResource.channel(area_id, channel_id)

    @staticmethod
    def _binding_address(
        scope: RoleBindingScope,
        current_resource: AccessResource,
    ) -> tuple[str, str]:
        if scope is RoleBindingScope.GLOBAL:
            return "", ""
        if current_resource.kind not in {
            AccessResourceKind.AREA,
            AccessResourceKind.CHANNEL,
        }:
            raise ValueError("Area and channel roles require a channel context")
        if scope is RoleBindingScope.AREA:
            return current_resource.area_id, ""
        if current_resource.kind is not AccessResourceKind.CHANNEL:
            raise ValueError("Channel roles require a channel context")
        return current_resource.area_id, current_resource.channel_id
