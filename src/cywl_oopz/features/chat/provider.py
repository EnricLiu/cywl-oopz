"""Provider contract and disabled implementation for text chat."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from cywl_oopz.core.errors import FeatureDisabledError

from .models import ChatChunk, ChatRequest, ChatResponse


class ChatProvider(Protocol):
    """Provider boundary; implementations must not expose vendor SDK objects."""

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Return one complete response."""

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Yield incremental output in order and then finish normally."""

    async def aclose(self) -> None:
        """Release provider resources during application shutdown."""


class DisabledChatProvider:
    """Keeps feature-disabled deployments explicit and safe."""

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Reject completion without accidentally making an external request."""
        del request
        raise FeatureDisabledError("Text chat is disabled")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        """Reject streaming without accidentally making an external request."""
        del request
        raise FeatureDisabledError("Text chat is disabled")
        yield ChatChunk()

    async def aclose(self) -> None:
        """Disabled providers own no resources."""
