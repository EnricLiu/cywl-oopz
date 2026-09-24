"""Trusted OOPZ mention extraction for Agent invocations."""

from __future__ import annotations

from typing import Any

from cywl_oopz.commands.models import CommandRequest
from cywl_oopz.conversation.models import ChatInvocation, ConversationKey


def conversation_key_from_context(context: Any) -> ConversationKey:
    """Project the SDK event location into a conversation key."""
    event = getattr(context, "event", None)
    message = getattr(event, "message", None)
    if message is None:
        raise ValueError("A chat command requires an OOPZ message event")
    person_id = str(getattr(message, "sender_id", "")).strip()
    if not person_id:
        raise ValueError("Message sender is required for a chat command")
    if bool(getattr(event, "is_private", False)):
        return ConversationKey(scope="private", area_id="", channel_id="", person_id=person_id)
    area_id = str(getattr(message, "area", "")).strip()
    channel_id = str(getattr(message, "channel", "")).strip()
    if not area_id or not channel_id:
        raise ValueError("Channel messages require area and channel identifiers")
    return ConversationKey(
        scope="channel",
        area_id=area_id,
        channel_id=channel_id,
        person_id=person_id,
    )


def invocation_from_context(
    context: Any,
    *,
    excluded_person_ids: tuple[str, ...] = (),
) -> ChatInvocation:
    """Extract stable message targets without exposing the SDK to features."""
    event = getattr(context, "event", None)
    message = getattr(event, "message", None)
    if message is None:
        raise ValueError("A chat invocation requires an OOPZ message event")
    sender_id = str(getattr(message, "sender_id", "")).strip()
    excluded = {sender_id, *(value.strip() for value in excluded_person_ids)}
    mentioned: list[str] = []
    for mention in getattr(message, "mention_list", ()) or ():
        person_id = str(getattr(mention, "person", "")).strip()
        if person_id and person_id not in excluded and person_id not in mentioned:
            mentioned.append(person_id)
    return ChatInvocation(
        source_message_id=str(getattr(message, "message_id", "")).strip(),
        transport_channel_id=str(getattr(message, "channel", "")).strip(),
        mentioned_person_ids=tuple(mentioned),
    )


class OopzChatInvocationFactory:
    """Exclude the bot and sender while preserving stable mentioned person IDs."""

    def __init__(self, bot_person_id: str) -> None:
        normalized = bot_person_id.strip()
        if not normalized:
            raise ValueError("OOPZ bot person ID must not be empty")
        self._bot_person_id = normalized

    def from_context(self, context: Any) -> ChatInvocation:
        return invocation_from_context(
            context,
            excluded_person_ids=(self._bot_person_id,),
        )

    def from_request(self, request: CommandRequest) -> ChatInvocation:
        excluded = {request.actor.person_id, self._bot_person_id}
        mentioned: list[str] = []
        for mention in request.mentions:
            if mention.person_id in excluded or mention.person_id in mentioned:
                continue
            mentioned.append(mention.person_id)
        return ChatInvocation(
            source_message_id=request.source.message_id,
            transport_channel_id=request.location.channel_id,
            mentioned_person_ids=tuple(mentioned),
        )
