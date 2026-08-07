"""Replaceable transport boundaries consumed by the shared audio core."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .models import DecodedAudioBlock, MasterPlaybackCursor


class MasterPcmOutput(Protocol):
    """The only owner allowed to control one SDK PCM output stream."""

    @property
    def cursor(self) -> MasterPlaybackCursor: ...

    async def write(self, pcm_s16le: bytes) -> MasterPlaybackCursor: ...

    async def flush(self) -> MasterPlaybackCursor: ...

    async def drain(self) -> MasterPlaybackCursor: ...

    async def aclose(self) -> None: ...


class AudioDecoder(Protocol):
    """Supervised canonical PCM decoder owned by one track playback."""

    def __aiter__(self) -> AsyncIterator[DecodedAudioBlock]: ...

    async def aclose(self) -> None: ...
