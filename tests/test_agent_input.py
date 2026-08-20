from __future__ import annotations

from oopz_sdk.models.message import Message

from cywl_oopz.features.agent.input import IMAGE_ONLY_PROMPT, AgentUserInput, ImageInputPart
from cywl_oopz.integrations.oopz.conversation_input import OopzConversationInputFactory


def _message(*, text: str, attachments: list[dict], mentions: list[dict] | None = None) -> Message:
    return Message.from_api(
        {
            "type": "TEXT",
            "area": "area-1",
            "channel": "channel-1",
            "person": "person-1",
            "content": text,
            "text": text,
            "mentionList": mentions or [],
            "attachments": attachments,
        }
    )


def _image(file_key: str = "/im/image.webp") -> dict:
    return {
        "attachmentType": "IMAGE",
        "fileKey": file_key,
        "url": "https://imimagecdn.oopz.cn" + file_key + "?sign=redacted",
        "fileSize": 1234,
        "hash": "hash-value",
        "width": 454,
        "height": 454,
        "animated": False,
    }


def test_oopz_adapter_preserves_text_image_order_and_excludes_mentions() -> None:
    message = _message(
        text=" (met)bot(met) hello  ![IMAGEw454h454](/im/image.webp) world",
        attachments=[_image()],
        mentions=[{"person": "bot"}],
    )

    result = OopzConversationInputFactory().from_message(message)

    assert result.text == "hello   world"
    assert [type(part).__name__ for part in result.parts] == [
        "TextInputPart",
        "ImageInputPart",
        "TextInputPart",
    ]
    image = result.images[0]
    assert image.source_file_key == "/im/image.webp"
    assert image.source_url.startswith("https://imimagecdn.oopz.cn/")
    assert image.resolved is False


def test_image_only_input_gets_a_stable_implicit_prompt() -> None:
    message = _message(
        text="![IMAGEw454h454](/im/image.webp)",
        attachments=[_image()],
    )

    result = OopzConversationInputFactory().from_message(message)

    assert result.text == ""
    assert result.implicit_prompt is True
    assert result.prompt == IMAGE_ONLY_PROMPT
    assert result.has_images is True


def test_agent_input_rejects_empty_content() -> None:
    try:
        AgentUserInput.from_parts(())
    except ValueError as exc:
        assert "text or an image" in str(exc)
    else:
        raise AssertionError("empty Agent input must be rejected")


def test_image_bytes_are_not_in_repr() -> None:
    image = ImageInputPart(data=b"secret", media_type="image/png", byte_size=6)

    assert "secret" not in repr(image)
