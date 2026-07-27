from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from oopz_sdk.exceptions import OopzApiError

from cywl_oopz.integrations.oopz.editable_messages import (
    EditableMessageRef,
    MessageAddress,
    OopzEditableMessageGateway,
)


@dataclass
class FakeResponse:
    status_code: int = 200
    payload: Any = None

    def json(self) -> Any:
        return self.payload if self.payload is not None else {"status": True, "data": True}


class FakeMessages:
    def __init__(self) -> None:
        self.signer = SimpleNamespace()
        self.sent: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []
        self.raw_requests: list[dict[str, Any]] = []
        self.response = FakeResponse()

    async def send_message(self, *texts: str, **kwargs: Any) -> Any:
        self.sent.append(("channel", texts, kwargs))
        return SimpleNamespace(message_id="channel-message", timestamp="123")

    async def send_private_message(self, *texts: str, **kwargs: Any) -> Any:
        self.sent.append(("private", texts, kwargs))
        return SimpleNamespace(message_id="private-message", timestamp="456")

    async def request_raw(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.raw_requests.append({"method": method, "url": url, **kwargs})
        return self.response


class FakeBot:
    def __init__(self) -> None:
        self.messages = FakeMessages()
        self.config = SimpleNamespace(
            base_url="https://gateway.example",
            get_headers=lambda: {},
        )


def address(scope: str = "channel") -> MessageAddress:
    return MessageAddress(
        scope=scope,
        area_id="area" if scope == "channel" else "",
        channel_id="channel",
        target_person_id="person" if scope == "private" else "",
        reference_message_id="source",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "kind", "message_id"),
    [
        ("channel", "channel", "channel-message"),
        ("private", "private", "private-message"),
    ],
)
async def test_create_reply_uses_the_correct_oopz_send_method(
    scope: str,
    kind: str,
    message_id: str,
) -> None:
    bot = FakeBot()
    gateway = OopzEditableMessageGateway(bot)

    result = await gateway.create_reply(address(scope), "✨ 正在准备回答…")

    assert result.message_id == message_id
    assert bot.messages.sent == [
        (
            kind,
            ("✨ 正在准备回答…",),
            {
                **({"target": "person"} if scope == "private" else {"area": "area"}),
                "channel": "channel",
                "reference_message_id": "source",
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "path"),
    [
        ("channel", "/im/session/v1/editGimMessage"),
        ("private", "/im/session/v1/editImMessage"),
    ],
)
async def test_replace_posts_the_confirmed_web_client_payload(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    path: str,
) -> None:
    bot = FakeBot()
    signed: list[tuple[Any, Any, str, str]] = []
    monkeypatch.setattr(
        "cywl_oopz.integrations.oopz.editable_messages.build_oopz_headers",
        lambda config, signer, request_path, body: (
            signed.append((config, signer, request_path, body)) or {"X-Test-Signature": "signed"}
        ),
    )
    gateway = OopzEditableMessageGateway(bot)
    ref = EditableMessageRef(
        message_id="message",
        timestamp="123",
        scope=scope,
        area_id="area" if scope == "channel" else "",
        channel_id="channel",
        target_person_id="person" if scope == "private" else "",
        reference_message_id="source",
    )

    await gateway.replace(ref, "**完成** ♪")

    request = bot.messages.raw_requests[0]
    body = json.loads(request["data"])
    assert request == {
        "method": "POST",
        "url": f"https://gateway.example{path}",
        "data": request["data"],
        "headers": {"X-Test-Signature": "signed"},
    }
    assert body == {
        "messageId": "message",
        "area": "area" if scope == "channel" else "",
        "channel": "channel",
        "target": "person" if scope == "private" else "",
        "clientMessageId": "",
        "timestamp": "123",
        "isMentionAll": False,
        "mentionList": [],
        "styleTags": [],
        "referenceMessageId": "source",
        "animated": False,
        "displayName": "",
        "duration": 0,
        "text": "**完成** ♪",
        "attachments": [],
        "changeAttachments": [],
    }
    assert signed[0][2:] == (path, json.dumps(body, separators=(",", ":"), ensure_ascii=False))


@pytest.mark.asyncio
async def test_replace_rejects_failed_oopz_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = FakeBot()
    bot.messages.response = FakeResponse(payload={"status": False, "error": "rejected"})
    monkeypatch.setattr(
        "cywl_oopz.integrations.oopz.editable_messages.build_oopz_headers",
        lambda *_: {},
    )
    gateway = OopzEditableMessageGateway(bot)
    ref = EditableMessageRef("message", "123", "channel", "area", "channel", "", "")

    with pytest.raises(OopzApiError, match="rejected"):
        await gateway.replace(ref, "new text")


def test_message_address_extracts_channel_and_private_contexts() -> None:
    message = SimpleNamespace(
        area="area",
        channel="channel",
        sender_id="person",
        message_id="source",
    )
    channel = SimpleNamespace(event=SimpleNamespace(message=message, is_private=False))
    private = SimpleNamespace(event=SimpleNamespace(message=message, is_private=True))

    assert MessageAddress.from_oopz_context(channel) == address("channel")
    assert MessageAddress.from_oopz_context(private) == address("private")
