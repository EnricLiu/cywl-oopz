from __future__ import annotations

import io

import httpx
import pytest
from PIL import Image

from cywl_oopz.core.errors import UserRequestError
from cywl_oopz.features.agent.input import AgentUserInput, ImageInputPart, TextInputPart
from cywl_oopz.features.agent.media import (
    AgentImagePolicy,
    AgentMediaIngestService,
    ImageContentValidator,
)
from cywl_oopz.features.chat.progress import ProgressKind
from cywl_oopz.integrations.oopz.image_loader import OopzImageContentLoader


def _png(*, size: tuple[int, int] = (2, 2), color: str = "red") -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


class StaticLoader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.calls: list[int] = []

    async def load(self, image: ImageInputPart, *, maximum_bytes: int) -> bytes:
        del image
        self.calls.append(maximum_bytes)
        return self.data


class RecordingProgress:
    def __init__(self) -> None:
        self.kinds: list[ProgressKind] = []

    async def emit(self, event) -> None:
        self.kinds.append(event.kind)


@pytest.mark.asyncio
async def test_ingest_resolves_images_in_order_and_reports_media_progress() -> None:
    loader = StaticLoader(_png())
    service = AgentMediaIngestService(loader, AgentImagePolicy(max_images=2))
    user_input = AgentUserInput.from_parts(
        [
            TextInputPart("看这张"),
            ImageInputPart(source_file_key="/one", source_url="https://imimagecdn.oopz.cn/one"),
            ImageInputPart(source_file_key="/two", source_url="https://imimagecdn.oopz.cn/two"),
        ]
    )
    progress = RecordingProgress()

    result = await service.ingest(user_input, progress)

    assert result.text == "看这张"
    assert result.resolved_images is True
    assert [image.media_type for image in result.images] == ["image/png", "image/png"]
    assert progress.kinds == [ProgressKind.MEDIA_LOADING, ProgressKind.MEDIA_READY]
    assert loader.calls == [8 * 1024 * 1024, 8 * 1024 * 1024]


@pytest.mark.asyncio
async def test_ingest_rejects_too_many_images_before_downloading() -> None:
    loader = StaticLoader(_png())
    service = AgentMediaIngestService(loader, AgentImagePolicy(max_images=1))
    user_input = AgentUserInput.from_parts(
        [
            ImageInputPart(source_file_key="/one"),
            ImageInputPart(source_file_key="/two"),
        ]
    )

    with pytest.raises(UserRequestError) as error:
        await service.ingest(user_input)

    assert error.value.code == "image_limit_exceeded"
    assert loader.calls == []


@pytest.mark.asyncio
async def test_validator_rejects_animated_gif() -> None:
    stream = io.BytesIO()
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    first.save(stream, format="GIF", save_all=True, append_images=[second], duration=10, loop=0)

    with pytest.raises(UserRequestError) as error:
        await ImageContentValidator(max_pixels=100).validate(stream.getvalue())

    assert error.value.code == "unsupported_image"


@pytest.mark.asyncio
async def test_oopz_loader_rejects_non_allowlisted_source_without_http_call() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=_png()))
    async with httpx.AsyncClient(transport=transport) as client:
        loader = OopzImageContentLoader(client)
        with pytest.raises(UserRequestError) as error:
            await loader.load(
                ImageInputPart(source_url="https://example.com/image.png"),
                maximum_bytes=1024,
            )

    assert error.value.code == "invalid_image_source"
