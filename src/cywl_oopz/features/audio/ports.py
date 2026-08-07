"""Replaceable transport boundaries consumed by the shared audio core."""

from __future__ import annotations

from typing import Protocol

from .models import MasterPlaybackCursor


class MasterPcmOutput(Protocol):
    """The only owner allowed to control one SDK PCM output stream."""

    @property
    def cursor(self) -> MasterPlaybackCursor: ...

    async def write(self, pcm_s16le: bytes) -> MasterPlaybackCursor: ...

    async def flush(self) -> MasterPlaybackCursor: ...

    async def drain(self) -> MasterPlaybackCursor: ...

    async def aclose(self) -> None: ...
