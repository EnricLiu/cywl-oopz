"""OOPZ adapters for reaction-triggered administration commands."""

from __future__ import annotations

import logging
from typing import Any

from oopz_sdk.utils.reaction_emoji import normalize_reaction_emoji

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.access.models import AccessPrincipal, AccessResource
from cywl_oopz.features.admin.models import (
    OopzMessageAddress,
    OopzMessageScope,
    OutboundMessageKind,
    OutboundMessageState,
)
from cywl_oopz.features.admin.reaction_commands import ReactionCommandInvocation

from .editable_messages import MessageAddress, OopzEditableMessageGateway

logger = logging.getLogger(__name__)


class OopzReactionCommandInvocationParser:
    """Project an SDK reaction event into trusted command values."""

    _addition_types = frozenset({"reply", "add", "added"})

    @classmethod
    def parse(cls, context: Any, event: Any) -> ReactionCommandInvocation | None:
        reaction_type = str(getattr(event, "type", "")).strip().casefold()
        if reaction_type not in cls._addition_types:
            return None
        actor_id = str(getattr(event, "person", "")).strip()
        bot_id = str(getattr(getattr(context, "config", None), "person_uid", "")).strip()
        message_id = str(getattr(event, "message_id", "")).strip()
        area_id = str(getattr(event, "area", "")).strip()
        channel_id = str(getattr(event, "channel", "")).strip()
        raw_emoji = str(getattr(event, "emoji", "")).strip()
        try:
            emoji = normalize_reaction_emoji(raw_emoji)
        except ValueError:
            emoji = raw_emoji
        if not actor_id or actor_id == bot_id or not message_id or not channel_id or not emoji:
            return None
        try:
            if area_id:
                address = OopzMessageAddress(
                    OopzMessageScope.CHANNEL,
                    area_id,
                    channel_id,
                )
                resource = AccessResource.channel(area_id, channel_id)
            else:
                address = OopzMessageAddress(
                    OopzMessageScope.PRIVATE,
                    "",
                    channel_id,
                    actor_id,
                )
                resource = AccessResource.private()
            return ReactionCommandInvocation(
                emoji,
                message_id,
                AccessPrincipal(actor_id),
                resource,
                address,
            )
        except ValueError as exc:
            logger.warning(
                "Ignored malformed OOPZ reaction command event: error=%s",
                exception_kind(exc),
            )
            return None


class OopzReactionCommandResponder:
    """Reply beside the reacted message and retain normal outbound tracking."""

    def __init__(self, messages: OopzEditableMessageGateway) -> None:
        self._messages = messages

    async def send(self, invocation: ReactionCommandInvocation, text: str) -> None:
        address = invocation.address
        created = await self._messages.create_reply(
            MessageAddress(
                scope=address.scope.value,
                area_id=address.area_id,
                channel_id=address.channel_id,
                target_person_id=address.target_person_id,
                reference_message_id=invocation.message_id,
                owner_person_id=invocation.principal.person_id,
            ),
            text,
        )
        await self._messages.track_created(
            created,
            kind=OutboundMessageKind.COMMAND_REPLY,
            state=OutboundMessageState.FINAL,
            owner_person_id=invocation.principal.person_id,
        )
        logger.debug(
            "Sent reaction command response: message=%s source=%s",
            opaque_ref(created.message_id),
            opaque_ref(invocation.message_id),
        )
