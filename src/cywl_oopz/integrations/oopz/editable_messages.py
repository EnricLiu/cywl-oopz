"""Narrow OOPZ gateway for creating and replacing one text message."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from oopz_sdk.auth.headers import build_oopz_headers
from oopz_sdk.exceptions import OopzApiError
from oopz_sdk.utils.payload import coerce_bool


@dataclass(frozen=True, slots=True)
class MessageAddress:
    """Transport address of the user message an Agent display replies to."""

    scope: str
    area_id: str
    channel_id: str
    target_person_id: str
    reference_message_id: str

    def __post_init__(self) -> None:
        if self.scope not in {"channel", "private"}:
            raise ValueError("Message scope must be 'channel' or 'private'")
        if not self.channel_id:
            raise ValueError("Message channel is required")
        if self.scope == "channel" and not self.area_id:
            raise ValueError("Channel messages require an area")
        if self.scope == "private" and not self.target_person_id:
            raise ValueError("Private messages require a target person")

    @classmethod
    def from_oopz_context(cls, context: Any) -> MessageAddress:
        """Extract stable addressing data at the OOPZ integration boundary."""
        event = getattr(context, "event", None)
        message = getattr(event, "message", None)
        if message is None:
            raise ValueError("An editable reply requires an OOPZ message event")
        is_private = bool(getattr(event, "is_private", False))
        return cls(
            scope="private" if is_private else "channel",
            area_id="" if is_private else str(getattr(message, "area", "")).strip(),
            channel_id=str(getattr(message, "channel", "")).strip(),
            target_person_id=(str(getattr(message, "sender_id", "")).strip() if is_private else ""),
            reference_message_id=str(getattr(message, "message_id", "")).strip(),
        )


@dataclass(frozen=True, slots=True)
class EditableMessageRef:
    """All metadata needed to replace a previously created OOPZ message."""

    message_id: str
    timestamp: str
    scope: str
    area_id: str
    channel_id: str
    target_person_id: str
    reference_message_id: str


class OopzEditableMessageGateway:
    """Use the confirmed OOPZ edit endpoints missing from oopz-sdk 0.13.1."""

    _CHANNEL_EDIT_PATH = "/im/session/v1/editGimMessage"
    _PRIVATE_EDIT_PATH = "/im/session/v1/editImMessage"

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    async def create_reply(
        self,
        address: MessageAddress,
        text: str,
    ) -> EditableMessageRef:
        """Create the sole display message and retain its edit metadata."""
        if address.scope == "private":
            result = await self._bot.messages.send_private_message(
                text,
                target=address.target_person_id,
                channel=address.channel_id,
                reference_message_id=address.reference_message_id or None,
            )
        else:
            result = await self._bot.messages.send_message(
                text,
                area=address.area_id,
                channel=address.channel_id,
                reference_message_id=address.reference_message_id or None,
            )
        return EditableMessageRef(
            message_id=str(result.message_id),
            timestamp=str(result.timestamp),
            scope=address.scope,
            area_id=address.area_id,
            channel_id=address.channel_id,
            target_person_id=address.target_person_id,
            reference_message_id=address.reference_message_id,
        )

    async def replace(self, message: EditableMessageRef, text: str) -> None:
        """Replace message text through the active edit endpoint."""
        if not text:
            raise ValueError("Replacement text must not be empty")
        path = self._PRIVATE_EDIT_PATH if message.scope == "private" else self._CHANNEL_EDIT_PATH
        body = self._edit_payload(message, text)
        encoded_body = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        headers = build_oopz_headers(
            self._bot.config,
            self._bot.messages.signer,
            path,
            encoded_body,
        )
        response = await self._bot.messages.request_raw(
            "POST",
            f"{self._bot.config.base_url}{path}",
            data=encoded_body.encode("utf-8"),
            headers=headers,
        )
        self._ensure_success(response)

    @staticmethod
    def _edit_payload(message: EditableMessageRef, text: str) -> dict[str, Any]:
        """Mirror the smallest stable payload emitted by the official Web client."""
        return {
            "messageId": message.message_id,
            "area": message.area_id,
            "channel": message.channel_id,
            "target": message.target_person_id,
            "clientMessageId": "",
            "timestamp": message.timestamp,
            "isMentionAll": False,
            "mentionList": [],
            "styleTags": [],
            "referenceMessageId": message.reference_message_id or None,
            "animated": False,
            "displayName": "",
            "duration": 0,
            "text": text,
            "attachments": [],
            "changeAttachments": [],
        }

    @staticmethod
    def _ensure_success(response: Any) -> None:
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise OopzApiError(
                "OOPZ edit response is not valid JSON",
                status_code=response.status_code,
                response=response,
            ) from exc
        if response.status_code != 200:
            raise OopzApiError(
                f"OOPZ edit request failed with HTTP {response.status_code}",
                status_code=response.status_code,
                payload=payload,
                response=response,
            )
        if not isinstance(payload, dict) or not coerce_bool(
            payload.get("status"),
            default=False,
        ):
            detail = "OOPZ edit request was rejected"
            if isinstance(payload, dict):
                for key in ("error", "message", "msg", "reason"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        detail = value.strip()
                        break
            raise OopzApiError(
                detail,
                status_code=response.status_code,
                payload=payload,
                response=response,
            )
