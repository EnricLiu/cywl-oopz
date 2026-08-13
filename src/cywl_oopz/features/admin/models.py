"""Persistence-facing values shared by administration integrations."""

from enum import StrEnum


class OopzMessageScope(StrEnum):
    CHANNEL = "channel"
    PRIVATE = "private"


class OutboundMessageKind(StrEnum):
    AGENT_RESPONSE = "agent_response"
    COMMAND_REPLY = "command_reply"
    STATUS = "status"
    NOTIFICATION = "notification"


class OutboundMessageState(StrEnum):
    ACTIVE = "active"
    FINAL = "final"
    RECALLED = "recalled"
    SUPERSEDED = "superseded"
