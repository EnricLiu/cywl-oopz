"""Safe aggregation for provider response streams."""

from __future__ import annotations

from collections.abc import AsyncIterator

from cywl_oopz.core.errors import ProviderResponseError

from .models import ChatChunk, ChatResponse


class StreamResponseAssembler:
    """Accumulates a provider stream without emitting one OOPZ message per token."""

    async def assemble(self, chunks: AsyncIterator[ChatChunk], fallback_model: str) -> ChatResponse:
        """Consume all chunks and reject streams that contain no textual output."""
        parts: list[str] = []
        model = fallback_model
        finish_reason = ""
        input_tokens: int | None = None
        output_tokens: int | None = None

        async for chunk in chunks:
            if chunk.delta:
                parts.append(chunk.delta)
            if chunk.model:
                model = chunk.model
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            if chunk.input_tokens is not None:
                input_tokens = chunk.input_tokens
            if chunk.output_tokens is not None:
                output_tokens = chunk.output_tokens

        content = "".join(parts)
        if not content.strip():
            raise ProviderResponseError("LLM stream contained no text")
        return ChatResponse(
            content=content,
            model=model,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
