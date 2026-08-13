"""Map trusted Agent identities into the shared scoped authorization service."""

from __future__ import annotations

from cywl_oopz.features.agent.models import AgentIdentity

from .models import AccessPrincipal, AccessResource, Permission
from .service import AuthorizationService


class AgentToolAuthorizationAdapter:
    """Perform fresh RBAC reads for one concrete Agent tool permission."""

    def __init__(self, authorization: AuthorizationService) -> None:
        self._authorization = authorization

    async def allows(self, identity: AgentIdentity, permission: Permission) -> bool:
        resource = self._resource(identity)
        if resource is None:
            return False
        return await self._authorization.allows(
            AccessPrincipal(identity.person_id),
            permission,
            resource,
        )

    @staticmethod
    def _resource(identity: AgentIdentity) -> AccessResource | None:
        key = identity.conversation
        if key.scope == "private":
            return AccessResource.private()
        if key.scope == "channel":
            return AccessResource.channel(key.area_id, key.channel_id)
        if key.area_id and identity.transport_channel_id:
            return AccessResource.channel(key.area_id, identity.transport_channel_id)
        if key.area_id:
            return AccessResource.area(key.area_id)
        return None
