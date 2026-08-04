"""Explicit placeholder used before a realtime Provider adapter is configured."""

from __future__ import annotations

from cywl_oopz.features.voice.errors import VoiceRuntimeUnavailableError
from cywl_oopz.features.voice.ports import VoiceSessionRuntimeContext


class UnavailableVoiceSessionRuntimeFactory:
    """Fail clearly instead of pretending that a silent fake session is live."""

    async def create(self, context: VoiceSessionRuntimeContext):
        del context
        raise VoiceRuntimeUnavailableError("No realtime voice Provider is configured")
