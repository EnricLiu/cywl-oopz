"""Narrow OOPZ gateway for creating and replacing one text message."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from oopz_sdk.auth.headers import build_oopz_headers
from oopz_sdk.exceptions import OopzApiError
from oopz_sdk.utils.payload import coerce_bool

from cywl_oopz.commands.models import CommandRequest, CommandScope
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.admin.models import (
    OopzMessageAddress,
    OopzMessageScope,
    OutboundMessageKind,
    OutboundMessageReceipt,
    OutboundMessageState,
)
from cywl_oopz.features.admin.ports import OutboundMessageRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MessageAddress:
    """Transport address of the user message an Agent display replies to."""

    scope: str
    area_id: str
    channel_id: str
    target_person_id: str
    reference_message_id: str
    owner_person_id: str = ""

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
            owner_person_id=str(getattr(message, "sender_id", "")).strip(),
        )

    @classmethod
    def from_command_request(cls, request: CommandRequest) -> MessageAddress:
        """Project one framework-neutral request into the OOPZ gateway model."""
        private = request.location.scope is CommandScope.PRIVATE
        return cls(
            scope="private" if private else "channel",
            area_id="" if private else request.location.area_id,
            channel_id=request.location.channel_id,
            target_person_id=(
                request.location.target_person_id or request.actor.person_id if private else ""
            ),
            reference_message_id=request.source.message_id,
            owner_person_id=request.actor.person_id,
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

    def __init__(
        self,
        bot: Any,
        receipts: OutboundMessageRepository | None = None,
    ) -> None:
        self._bot = bot
        self._receipts = receipts

    async def create_reply(
        self,
        address: MessageAddress,
        text: str,
    ) -> EditableMessageRef:
        """Create the sole display message and retain its edit metadata."""
        address_ref = self._address_ref(address)
        logger.debug(
            "Creating OOPZ editable reply: address=%s scope=%s characters=%s",
            address_ref,
            address.scope,
            len(text),
        )
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
        message = EditableMessageRef(
            message_id=str(result.message_id),
            timestamp=str(result.timestamp),
            scope=address.scope,
            area_id=address.area_id,
            channel_id=address.channel_id,
            target_person_id=address.target_person_id,
            reference_message_id=address.reference_message_id,
        )
        logger.info(
            "Created OOPZ editable reply: address=%s scope=%s",
            address_ref,
            address.scope,
        )
        return message

    async def track_created(
        self,
        message: EditableMessageRef,
        *,
        kind: OutboundMessageKind,
        state: OutboundMessageState,
        owner_person_id: str = "",
    ) -> None:
        """Best-effort tracking kept separate from transport message creation."""
        await self._track(message, kind, state, owner_person_id)

    async def bind_agent_run(self, message: EditableMessageRef, run_id: Any) -> None:
        if self._receipts is None:
            return
        try:
            bound = await self._receipts.bind_agent_run(message.message_id, run_id)
            if not bound:
                logger.warning(
                    "Could not bind outbound message to Agent run: message=%s",
                    opaque_ref(message.message_id),
                )
        except Exception as exc:
            logger.warning(
                "Outbound Agent run binding degraded: message=%s error=%s",
                opaque_ref(message.message_id),
                type(exc).__name__,
            )

    async def finalize(
        self,
        message: EditableMessageRef,
        snapshot: dict[str, object],
    ) -> None:
        if self._receipts is None:
            return
        try:
            updated = await self._receipts.update_state(
                message.message_id,
                OutboundMessageState.FINAL,
                diagnostic_snapshot=snapshot,
            )
            if not updated:
                logger.warning(
                    "Could not finalize outbound message receipt: message=%s",
                    opaque_ref(message.message_id),
                )
        except Exception as exc:
            logger.warning(
                "Outbound message finalization degraded: message=%s error=%s",
                opaque_ref(message.message_id),
                type(exc).__name__,
            )

    async def supersede(
        self,
        message: EditableMessageRef,
        snapshot: dict[str, object],
    ) -> None:
        if self._receipts is None:
            return
        try:
            await self._receipts.update_state(
                message.message_id,
                OutboundMessageState.SUPERSEDED,
                diagnostic_snapshot=snapshot,
            )
        except Exception as exc:
            logger.warning(
                "Outbound message supersede degraded: message=%s error=%s",
                opaque_ref(message.message_id),
                type(exc).__name__,
            )

    async def promote_agent_response(
        self,
        message: EditableMessageRef,
        run_id: Any,
        snapshot: dict[str, object],
    ) -> None:
        if self._receipts is None:
            return
        try:
            updated = await self._receipts.promote_agent_response(
                message.message_id,
                run_id,
                snapshot,
            )
            if not updated:
                logger.warning(
                    "Could not promote direct Agent response: message=%s",
                    opaque_ref(message.message_id),
                )
        except Exception as exc:
            logger.warning(
                "Direct Agent response promotion degraded: message=%s error=%s",
                opaque_ref(message.message_id),
                type(exc).__name__,
            )

    async def replace(self, message: EditableMessageRef, text: str) -> None:
        """Replace message text through the active edit endpoint."""
        if not text:
            raise ValueError("Replacement text must not be empty")
        logger.debug(
            "Replacing OOPZ editable reply: message=%s scope=%s characters=%s",
            opaque_ref(message.message_id, message.timestamp),
            message.scope,
            len(text),
        )
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
        logger.debug(
            "Replaced OOPZ editable reply: message=%s",
            opaque_ref(message.message_id, message.timestamp),
        )

    @staticmethod
    def _address_ref(address: MessageAddress) -> str:
        return opaque_ref(
            address.scope,
            address.area_id,
            address.channel_id,
            address.target_person_id,
            address.reference_message_id,
        )

    async def _track(
        self,
        message: EditableMessageRef,
        kind: OutboundMessageKind,
        state: OutboundMessageState,
        owner_person_id: str,
    ) -> None:
        if self._receipts is None:
            return
        try:
            await self._receipts.create(
                OutboundMessageReceipt(
                    message_id=message.message_id,
                    message_timestamp=message.timestamp,
                    kind=kind,
                    state=state,
                    address=OopzMessageAddress(
                        OopzMessageScope(message.scope),
                        message.area_id,
                        message.channel_id,
                        message.target_person_id,
                    ),
                    in_reply_to_message_id=message.reference_message_id,
                    owner_person_id=owner_person_id,
                )
            )
        except Exception as exc:
            logger.warning(
                "Editable message tracking degraded: message=%s error=%s",
                opaque_ref(message.message_id),
                type(exc).__name__,
            )

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
