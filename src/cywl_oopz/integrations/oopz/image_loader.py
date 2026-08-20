"""HTTP implementation for OOPZ CDN image input."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

import httpx

from cywl_oopz.core.errors import UserRequestError
from cywl_oopz.features.agent.input import ImageInputPart


class OopzImageContentLoader:
    """Download only configured OOPZ image CDN URLs with a hard byte budget."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        allowed_hosts: Iterable[str] = ("imimagecdn.oopz.cn",),
    ) -> None:
        self._client = client
        self._allowed_hosts = frozenset(host.casefold() for host in allowed_hosts if host.strip())
        if not self._allowed_hosts:
            raise ValueError("At least one OOPZ image host must be allowed")

    async def load(self, image: ImageInputPart, *, maximum_bytes: int) -> bytes:
        """Read a response incrementally; signed URLs never enter error messages."""
        url = image.source_url
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname is None:
            raise UserRequestError("invalid_image_source", "这张图片的来源无效，请重新发送。")
        if parsed.hostname.casefold() not in self._allowed_hosts:
            raise UserRequestError("invalid_image_source", "这张图片的来源无效，请重新发送。")
        try:
            async with self._client.stream("GET", url, follow_redirects=False) as response:
                if response.is_redirect or response.status_code >= 400:
                    raise UserRequestError(
                        "image_download_failed",
                        "图片暂时无法读取，请重新发送或稍后再试。",
                    )
                content_length = response.headers.get("content-length")
                if content_length is not None and int(content_length) > maximum_bytes:
                    raise UserRequestError(
                        "image_too_large", "图片文件太大了，请换一张更小的图片。"
                    )
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > maximum_bytes:
                        raise UserRequestError(
                            "image_too_large",
                            "图片文件太大了，请换一张更小的图片。",
                        )
                if not chunks:
                    raise UserRequestError(
                        "image_download_failed",
                        "图片暂时无法读取，请重新发送或稍后再试。",
                    )
                return bytes(chunks)
        except UserRequestError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise UserRequestError(
                "image_download_failed",
                "图片暂时无法读取，请重新发送或稍后再试。",
            ) from exc
