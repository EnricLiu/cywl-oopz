"""Transport-neutral no-op status surface for optional live presentation."""

from __future__ import annotations

from .models import VoiceSessionStatus


class NoopVoiceSessionStatusSink:
    @property
    def owns_message(self) -> bool:
        return False

    def emit(self, status: VoiceSessionStatus) -> None:
        del status

    async def aclose(self) -> None:
        return
