"""Pydantic AI adapter behind the project-owned AgentEngine port."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError
from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    BinaryContent,
    ModelAPIError,
    ModelHTTPError,
    ModelRetry,
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

from cywl_oopz.core.errors import (
    AgentInternalError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.chat.progress import ProgressSink, emit_progress

from .input import ImageInputPart, TextInputPart
from .models import AgentMessage, AgentRunRequest, AgentRunResult, AgentStopReason
from .progress import ConversationToolProgressReporter, PydanticAiProgressMapper
from .provider_retry import bind_provider_retry_progress
from .registry import AgentModelRegistry
from .tools.models import ToolCall, ToolDescriptor, ToolExecutionContext
from .tools.ports import AgentToolRuntime

logger = logging.getLogger(__name__)

_RAW_TOOL_ARGUMENTS = "__cywl_raw_tool_arguments__"
_MALFORMED_TOOL_ARGUMENTS = "__cywl_malformed_tool_arguments__"
_MAX_ARGUMENT_ISSUES = 5


class _ToolArgumentsEnvelopeValidator:
    """Decode any model JSON into kwargs without trusting its root shape."""

    def validate_json(
        self,
        value: str | bytes | bytearray,
        *,
        allow_partial: object = None,
        context: object = None,
    ) -> dict[str, object]:
        del allow_partial, context
        try:
            decoded = json.loads(value or "{}")
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return {
                _RAW_TOOL_ARGUMENTS: None,
                _MALFORMED_TOOL_ARGUMENTS: True,
            }
        return {
            _RAW_TOOL_ARGUMENTS: decoded,
            _MALFORMED_TOOL_ARGUMENTS: False,
        }

    def validate_python(
        self,
        value: object,
        *,
        allow_partial: object = None,
        context: object = None,
    ) -> dict[str, object]:
        del allow_partial, context
        return {
            _RAW_TOOL_ARGUMENTS: value,
            _MALFORMED_TOOL_ARGUMENTS: False,
        }


@dataclass(frozen=True, slots=True)
class _ToolArgumentValidation:
    arguments: dict[str, object] | None
    issues: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.arguments is not None


@dataclass(slots=True)
class _ToolRunDependencies:
    runtime: AgentToolRuntime
    context: ToolExecutionContext
    semaphore: asyncio.Semaphore
    progress: ProgressSink | None


class PydanticAiAgentEngine:
    """Execute bounded runs while keeping framework types inside this adapter."""

    def __init__(
        self,
        registry: AgentModelRegistry,
        tools: AgentToolRuntime | None = None,
    ) -> None:
        self._registry = registry
        self._tools = tools

    async def run(
        self,
        request: AgentRunRequest,
        progress: ProgressSink | None = None,
    ) -> AgentRunResult:
        """Run an Agent loop with hard wall-clock, model, token, and tool budgets."""
        run_ref = opaque_ref("agent-run", request.run_id)
        logger.info(
            "Agent engine run started: run_ref=%s model=%s/%s context_messages=%s "
            "tools=%s timeout_seconds=%s",
            run_ref,
            request.model.provider_alias,
            request.model.model_alias,
            len(request.context),
            len(request.enabled_tools),
            request.limits.timeout_seconds,
        )
        try:
            model = await self._registry.model(request.model)
            instructions, history = self._map_context(request.context)
            framework_tools, dependencies, descriptors = self._build_tools(request, progress)
            progress_mapper = PydanticAiProgressMapper(descriptors)
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
        except asyncio.CancelledError:
            raise
        except ProviderError:
            raise
        except Exception as exc:
            logger.exception(
                "Agent engine bootstrap failed: run_ref=%s phase=engine_bootstrap "
                "responsibility=internal recoverability=terminal code=agent_internal error=%s",
                run_ref,
                exception_kind(exc),
            )
            raise AgentInternalError("Agent engine bootstrap failed") from exc
        try:
            with bind_provider_retry_progress(progress):
                await emit_progress(progress, progress_mapper.thinking())
                async with asyncio.timeout(request.limits.timeout_seconds):
                    if dependencies is None:
                        event_stream = agent.run_stream_events(
                            self._current_input_content(request),
                            message_history=history,
                            usage_limits=usage_limits,
                        )
                    else:
                        event_stream = agent.run_stream_events(
                            self._current_input_content(request),
                            deps=dependencies,
                            message_history=history,
                            usage_limits=usage_limits,
                        )
                    result = None
                    async with event_stream as events:
                        async for framework_event in events:
                            if isinstance(framework_event, AgentRunResultEvent):
                                result = framework_event.result
                            for mapped_event in progress_mapper.map(framework_event):
                                await emit_progress(progress, mapped_event)
                    if result is None:
                        raise ProviderResponseError("Agent stream ended without a result")
        except TimeoutError as exc:
            logger.warning(
                "Agent engine timed out: run_ref=%s phase=provider "
                "responsibility=dependency recoverability=terminal code=provider_timeout error=%s",
                run_ref,
                exception_kind(exc),
            )
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
            exhausted = AgentRunResult(
                output="本次对话已达到运行预算，请缩短问题或开始新的对话。",
                stop_reason=reason,
            )
            logger.warning(
                "Agent engine reached usage limit: run_ref=%s phase=provider "
                "responsibility=request recoverability=terminal code=usage_limit reason=%s",
                run_ref,
                reason.value,
            )
            return exhausted
        except (ModelAPIError, ModelHTTPError) as exc:
            logger.warning(
                "Agent model request failed: run_ref=%s phase=provider "
                "responsibility=dependency recoverability=terminal code=provider_error error=%s",
                run_ref,
                exception_kind(exc),
            )
            raise ProviderError("Agent model request failed") from exc
        except (UnexpectedModelBehavior, UserError) as exc:
            logger.warning(
                "Agent model response invalid: run_ref=%s phase=provider "
                "responsibility=model recoverability=terminal code=invalid_provider_response "
                "error=%s",
                run_ref,
                exception_kind(exc),
            )
            raise ProviderResponseError("Agent model returned an invalid response") from exc
        except ProviderError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Agent engine stream failed internally: run_ref=%s phase=engine_stream "
                "responsibility=internal recoverability=terminal code=agent_internal error=%s",
                run_ref,
                exception_kind(exc),
            )
            raise AgentInternalError("Agent engine stream failed") from exc

        try:
            output = result.output
            if not isinstance(output, str) or not output.strip():
                raise ProviderResponseError("Agent model returned no text")
            usage = result.usage
            intermediate_messages = self._map_new_tool_messages(
                result.new_messages(),
                descriptors,
            )
        except ProviderError:
            raise
        except Exception as exc:
            logger.exception(
                "Agent engine result mapping failed: run_ref=%s phase=result_mapping "
                "responsibility=internal recoverability=terminal code=agent_internal error=%s",
                run_ref,
                exception_kind(exc),
            )
            raise AgentInternalError("Agent engine result mapping failed") from exc
        logger.info(
            "Agent engine run completed: run_ref=%s model_requests=%s tool_calls=%s "
            "input_tokens=%s output_tokens=%s",
            run_ref,
            usage.requests,
            usage.tool_calls,
            usage.input_tokens,
            usage.output_tokens,
        )
        return AgentRunResult(
            output=output.strip(),
            stop_reason=AgentStopReason.COMPLETED,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model_requests=usage.requests,
            tool_calls=usage.tool_calls,
            intermediate_messages=intermediate_messages,
        )

    async def aclose(self) -> None:
        """Close registry-owned clients."""
        logger.debug("Closing Agent engine")
        await self._registry.aclose()

    @staticmethod
    def _current_input_content(request: AgentRunRequest) -> str | list[str | BinaryContent]:
        user_input = request.user_input
        if user_input is None:
            return request.prompt
        if user_input.has_images and not user_input.resolved_images:
            raise AgentInternalError("Agent image input reached the provider unresolved")
        content: list[str | BinaryContent] = []
        if user_input.implicit_prompt:
            content.append(user_input.prompt)
        for part in user_input.parts:
            if isinstance(part, TextInputPart):
                content.append(part.text)
            elif isinstance(part, ImageInputPart):
                if part.data is None:
                    raise AgentInternalError("Agent image input has no runtime bytes")
                content.append(BinaryContent(data=part.data, media_type=part.media_type))
        return content or request.prompt

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
        progress: ProgressSink | None,
    ) -> tuple[
        list[Tool[_ToolRunDependencies]],
        _ToolRunDependencies | None,
        tuple[ToolDescriptor, ...],
    ]:
        if not request.enabled_tools:
            return [], None, ()
        if self._tools is None:
            raise AgentInternalError("Agent tools are not configured")
        descriptors = self._tools.descriptors(request.enabled_tools)
        if tuple(descriptor.name for descriptor in descriptors) != tuple(
            sorted(request.enabled_tools)
        ):
            raise AgentInternalError("Agent tool set changed before execution")

        context = ToolExecutionContext(
            run_id=request.run_id,
            identity=request.identity,
            limits=request.limits,
            enabled_tools=request.enabled_tools,
            skill_scope=request.skill_scope,
        )
        dependencies = _ToolRunDependencies(
            runtime=self._tools,
            context=context,
            semaphore=asyncio.Semaphore(request.limits.max_parallel_tools),
            progress=progress,
        )
        framework_tools = [
            self._framework_tool(descriptor.name, descriptor, dependencies)
            for descriptor in descriptors
        ]
        return framework_tools, dependencies, descriptors

    @staticmethod
    def _framework_tool(
        name: str,
        descriptor: ToolDescriptor,
        dependencies: _ToolRunDependencies,
    ) -> Tool[_ToolRunDependencies]:
        def validate_arguments(
            context: RunContext[_ToolRunDependencies],
            **envelope: Any,
        ) -> None:
            validation = PydanticAiAgentEngine._validate_tool_arguments(
                descriptor,
                envelope,
            )
            if validation.valid:
                return
            logger.warning(
                "Agent tool arguments rejected: run_ref=%s call_ref=%s tool=%s "
                "phase=tool_validate responsibility=model recoverability=retry "
                "code=invalid_arguments attempt=%s max_retries=%s issues=%s",
                opaque_ref("agent-run", dependencies.context.run_id),
                opaque_ref(
                    "agent-tool-call",
                    dependencies.context.run_id,
                    context.tool_call_id or "missing",
                ),
                name,
                context.retry + 1,
                descriptor.max_retries,
                ",".join(validation.issues),
            )
            if context.retry < descriptor.max_retries:
                raise ModelRetry(PydanticAiAgentEngine._argument_retry_message(validation.issues))

        async def invoke(
            context: RunContext[_ToolRunDependencies],
            **arguments: Any,
        ) -> dict[str, object]:
            call_id = context.tool_call_id
            if not call_id:
                return {"ok": False, "error": "missing_tool_call_id"}
            validation = PydanticAiAgentEngine._validate_tool_arguments(
                descriptor,
                arguments,
            )
            if not validation.valid:
                return {"ok": False, "error": "invalid_arguments"}
            execution_context = replace(
                dependencies.context,
                model_requests_used=context.usage.requests,
                tool_calls_used=context.usage.tool_calls,
                progress=(
                    ConversationToolProgressReporter(
                        dependencies.progress,
                        call_id=call_id,
                        tool_name=name,
                        tool_display_name=descriptor.display_name,
                    )
                    if dependencies.progress is not None
                    else None
                ),
            )
            async with dependencies.semaphore:
                result = await dependencies.runtime.execute(
                    ToolCall(call_id, name, validation.arguments),
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
            args_validator=validate_arguments,
        )
        # Tool.from_schema() deliberately uses any_schema() at runtime. Replace it with
        # a root-shape-safe envelope so scalar, list, null, and malformed JSON never
        # reach invoke() through ``**arguments`` or expose their original values.
        tool.function_schema.validator = _ToolArgumentsEnvelopeValidator()  # type: ignore[assignment]
        tool.max_retries = descriptor.max_retries
        return tool

    @staticmethod
    def _validate_tool_arguments(
        descriptor: ToolDescriptor,
        envelope: dict[str, Any],
    ) -> _ToolArgumentValidation:
        if envelope.get(_MALFORMED_TOOL_ARGUMENTS) is True:
            return _ToolArgumentValidation(None, ("$: malformed_json",))
        raw_arguments = envelope.get(_RAW_TOOL_ARGUMENTS)
        try:
            validated = descriptor.input_model.model_validate(raw_arguments)
        except ValidationError as exc:
            issues: list[str] = []
            for error in exc.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )[:_MAX_ARGUMENT_ISSUES]:
                location = ".".join(str(item) for item in error["loc"]) or "$"
                issues.append(f"{location}: {error['type']}")
            return _ToolArgumentValidation(
                None,
                tuple(issues) or ("$: invalid_arguments",),
            )
        return _ToolArgumentValidation(
            validated.model_dump(mode="json"),
            (),
        )

    @staticmethod
    def _argument_retry_message(issues: tuple[str, ...]) -> str:
        detail = "; ".join(issues[:_MAX_ARGUMENT_ISSUES])
        return f"工具参数不符合已提供的 JSON Schema，请修正后重试。问题：{detail}"

    @staticmethod
    def _map_new_tool_messages(
        messages: list[ModelMessage],
        descriptors: tuple[ToolDescriptor, ...],
    ) -> tuple[AgentMessage, ...]:
        ephemeral_tools = frozenset(
            descriptor.name for descriptor in descriptors if not descriptor.replay_in_history
        )
        mapped: list[AgentMessage] = []
        for message in messages:
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart) and part.tool_name not in ephemeral_tools:
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
                    if isinstance(part, ToolReturnPart) and part.tool_name not in ephemeral_tools:
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
