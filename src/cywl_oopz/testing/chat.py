"""In-memory chat ports that never perform network or database I/O."""

from __future__ import annotations

from collections.abc import AsyncIterator

from cywl_oopz.features.chat.models import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    ConversationKey,
    ConversationSession,
)


class InMemoryConversationRepository:
    """A deterministic repository fake that copies no external state."""

    def __init__(self) -> None:
        self.sessions: dict[ConversationKey, ConversationSession] = {}

    async def get(self, key: ConversationKey) -> ConversationSession | None:
        return self.sessions.get(key)

    async def save(self, session: ConversationSession) -> None:
        self.sessions[session.key] = session

    async def delete(self, key: ConversationKey) -> None:
        self.sessions.pop(key, None)


class RecordingChatProvider:
    """Returns supplied text while retaining requests for assertions."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.requests: list[ChatRequest] = []
        self._responses = responses or ["test response"]

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(content=self._next_response(), model=request.model)

    async def _stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.requests.append(request)
        yield ChatChunk(delta=self._next_response(), model=request.model, finish_reason="stop")

    def stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        return self._stream(request)

    async def aclose(self) -> None:
        return None

    def _next_response(self) -> str:
        if not self._responses:
            return "test response"
        return self._responses.pop(0)
