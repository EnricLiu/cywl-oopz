"""Pydantic AI adapter behind the project-owned AgentEngine port."""

from __future__ import annotations

import asyncio

from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UsageLimits,
    UserError,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from cywl_oopz.core.errors import ProviderError, ProviderResponseError, ProviderTimeoutError

from .models import AgentMessage, AgentRunRequest, AgentRunResult, AgentStopReason
from .registry import AgentModelRegistry


class PydanticAiAgentEngine:
    """Execute bounded runs while keeping framework types inside this adapter."""

    def __init__(self, registry: AgentModelRegistry) -> None:
        self._registry = registry

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        """Run a no-tool Agent loop with hard wall-clock and usage budgets."""
        model = await self._registry.model(request.model)
        instructions, history = self._map_context(request.context)
        agent = Agent(model, instructions=instructions)
        usage_limits = UsageLimits(
            request_limit=request.limits.max_model_requests,
            tool_calls_limit=request.limits.max_tool_calls,
            total_tokens_limit=request.limits.max_total_tokens,
        )
        try:
            async with asyncio.timeout(request.limits.timeout_seconds):
                result = await agent.run(
                    request.prompt,
                    message_history=history,
                    usage_limits=usage_limits,
                )
        except TimeoutError as exc:
            raise ProviderTimeoutError("Agent run timed out") from exc
        except UsageLimitExceeded as exc:
            message = str(exc)
            reason = (
                AgentStopReason.TOKEN_LIMIT
                if "token" in message
                else AgentStopReason.MODEL_REQUEST_LIMIT
            )
            return AgentRunResult(
                output="本次对话已达到运行预算，请缩短问题或开始新的对话。",
                stop_reason=reason,
            )
        except (ModelAPIError, ModelHTTPError) as exc:
            raise ProviderError("Agent model request failed") from exc
        except (UnexpectedModelBehavior, UserError) as exc:
            raise ProviderResponseError("Agent model returned an invalid response") from exc

        output = result.output
        if not isinstance(output, str) or not output.strip():
            raise ProviderResponseError("Agent model returned no text")
        usage = result.usage
        return AgentRunResult(
            output=output.strip(),
            stop_reason=AgentStopReason.COMPLETED,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model_requests=usage.requests,
            tool_calls=usage.tool_calls,
        )

    async def aclose(self) -> None:
        """Close registry-owned clients."""
        await self._registry.aclose()

    @staticmethod
    def _map_context(
        context: tuple[AgentMessage, ...],
    ) -> tuple[str | None, list[ModelMessage]]:
        instructions: list[str] = []
        messages: list[ModelMessage] = []
        for message in context:
            text = message.content.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if message.role == "system":
                instructions.append(text)
            elif message.role == "user":
                messages.append(ModelRequest(parts=[UserPromptPart(text)]))
            elif message.role == "assistant":
                messages.append(ModelResponse(parts=[TextPart(text)]))
        return ("\n\n".join(instructions) or None), messages
