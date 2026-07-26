"""Pydantic AI adapter behind the project-owned AgentEngine port."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelHTTPError,
    RunContext,
    Tool,
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
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from cywl_oopz.core.errors import ProviderError, ProviderResponseError, ProviderTimeoutError

from .models import AgentMessage, AgentRunRequest, AgentRunResult, AgentStopReason
from .registry import AgentModelRegistry
from .tools.models import ToolCall, ToolDescriptor, ToolExecutionContext
from .tools.ports import AgentToolRuntime


@dataclass(slots=True)
class _ToolRunDependencies:
    runtime: AgentToolRuntime
    context: ToolExecutionContext
    semaphore: asyncio.Semaphore


class PydanticAiAgentEngine:
    """Execute bounded runs while keeping framework types inside this adapter."""

    def __init__(
        self,
        registry: AgentModelRegistry,
        tools: AgentToolRuntime | None = None,
    ) -> None:
        self._registry = registry
        self._tools = tools

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        """Run an Agent loop with hard wall-clock, model, token, and tool budgets."""
        model = await self._registry.model(request.model)
        instructions, history = self._map_context(request.context)
        framework_tools, dependencies = self._build_tools(request)
        if framework_tools:
            agent = Agent(
                model,
                instructions=instructions,
                deps_type=_ToolRunDependencies,
                tools=framework_tools,
            )
        else:
            agent = Agent(model, instructions=instructions)
        usage_limits = UsageLimits(
            request_limit=request.limits.max_model_requests,
            tool_calls_limit=request.limits.max_tool_calls,
            total_tokens_limit=request.limits.max_total_tokens,
        )
        try:
            async with asyncio.timeout(request.limits.timeout_seconds):
                if dependencies is None:
                    result = await agent.run(
                        request.prompt,
                        message_history=history,
                        usage_limits=usage_limits,
                    )
                else:
                    result = await agent.run(
                        request.prompt,
                        deps=dependencies,
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
                else (
                    AgentStopReason.TOOL_CALL_LIMIT
                    if "tool_call" in message
                    else AgentStopReason.MODEL_REQUEST_LIMIT
                )
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
            intermediate_messages=self._map_new_tool_messages(result.new_messages()),
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
        pending_request_parts: list[UserPromptPart | ToolReturnPart] = []
        pending_response_parts: list[TextPart | ToolCallPart] = []

        def flush_request() -> None:
            if pending_request_parts:
                messages.append(ModelRequest(parts=list(pending_request_parts)))
                pending_request_parts.clear()

        def flush_response() -> None:
            if pending_response_parts:
                messages.append(ModelResponse(parts=list(pending_response_parts)))
                pending_response_parts.clear()

        for message in context:
            text = message.content.get("text")
            if message.role == "system":
                if isinstance(text, str) and text.strip():
                    instructions.append(text)
                continue
            if message.role == "user" and message.kind == "text":
                if not isinstance(text, str) or not text.strip():
                    continue
                flush_response()
                pending_request_parts.append(UserPromptPart(text))
            elif message.role == "assistant" and message.kind == "text":
                if not isinstance(text, str) or not text.strip():
                    continue
                flush_request()
                pending_response_parts.append(TextPart(text))
            elif message.role == "assistant" and message.kind == "tool_call":
                flush_request()
                tool_name = message.content.get("tool_name")
                call_id = message.content.get("tool_call_id")
                arguments = message.content.get("arguments")
                if isinstance(tool_name, str) and isinstance(call_id, str):
                    pending_response_parts.append(
                        ToolCallPart(
                            tool_name,
                            arguments if isinstance(arguments, dict | str) else {},
                            call_id,
                        )
                    )
            elif message.role == "tool" and message.kind == "tool_result":
                flush_response()
                tool_name = message.content.get("tool_name")
                call_id = message.content.get("tool_call_id")
                result = message.content.get("result")
                if isinstance(tool_name, str) and isinstance(call_id, str):
                    pending_request_parts.append(ToolReturnPart(tool_name, result, call_id))
        flush_request()
        flush_response()
        return ("\n\n".join(instructions) or None), messages

    def _build_tools(
        self,
        request: AgentRunRequest,
    ) -> tuple[list[Tool[_ToolRunDependencies]], _ToolRunDependencies | None]:
        if not request.enabled_tools:
            return [], None
        if self._tools is None:
            raise ProviderResponseError("Agent tools are not configured")
        descriptors = self._tools.descriptors(request.enabled_tools)
        if tuple(descriptor.name for descriptor in descriptors) != tuple(
            sorted(request.enabled_tools)
        ):
            raise ProviderResponseError("Agent tool set changed before execution")

        context = ToolExecutionContext(
            run_id=request.run_id,
            identity=request.identity,
            limits=request.limits,
            enabled_tools=request.enabled_tools,
        )
        dependencies = _ToolRunDependencies(
            runtime=self._tools,
            context=context,
            semaphore=asyncio.Semaphore(request.limits.max_parallel_tools),
        )
        framework_tools = [
            self._framework_tool(descriptor.name, descriptor, dependencies)
            for descriptor in descriptors
        ]
        return framework_tools, dependencies

    @staticmethod
    def _framework_tool(
        name: str,
        descriptor: ToolDescriptor,
        dependencies: _ToolRunDependencies,
    ) -> Tool[_ToolRunDependencies]:
        async def invoke(
            context: RunContext[_ToolRunDependencies],
            **arguments: Any,
        ) -> dict[str, object]:
            call_id = context.tool_call_id
            if not call_id:
                return {"ok": False, "error": "missing_tool_call_id"}
            execution_context = replace(
                dependencies.context,
                model_requests_used=context.usage.requests,
                tool_calls_used=context.usage.tool_calls,
            )
            async with dependencies.semaphore:
                result = await dependencies.runtime.execute(
                    ToolCall(call_id, name, arguments),
                    execution_context,
                )
            return result.model_payload()

        tool = Tool.from_schema(
            invoke,
            name=name,
            description=descriptor.description,
            json_schema=descriptor.input_model.model_json_schema(),
            takes_ctx=True,
            sequential=descriptor.sequential,
        )
        tool.max_retries = descriptor.max_retries
        return tool

    @staticmethod
    def _map_new_tool_messages(
        messages: list[ModelMessage],
    ) -> tuple[AgentMessage, ...]:
        mapped: list[AgentMessage] = []
        for message in messages:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        mapped.append(
                            AgentMessage(
                                "assistant",
                                "tool_call",
                                {
                                    "version": 1,
                                    "tool_call_id": part.tool_call_id,
                                    "tool_name": part.tool_name,
                                    "arguments": part.args,
                                },
                            )
                        )
            elif isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, ToolReturnPart):
                        mapped.append(
                            AgentMessage(
                                "tool",
                                "tool_result",
                                {
                                    "version": 1,
                                    "tool_call_id": part.tool_call_id,
                                    "tool_name": part.tool_name,
                                    "result": part.content,
                                    "outcome": part.outcome,
                                },
                            )
                        )
        return tuple(mapped)
