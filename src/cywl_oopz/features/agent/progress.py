"""Map Pydantic AI stream events to project-owned display-safe progress."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolReturnPart,
)

from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind

from .tool_progress import ToolProgressFormatter
from .tools.models import ToolDescriptor


class PydanticAiProgressMapper:
    """Stateful mapper for model turns without exposing reasoning or tool data."""

    def __init__(
        self,
        descriptors: tuple[ToolDescriptor, ...],
        details: ToolProgressFormatter | None = None,
    ) -> None:
        self._display_names = {
            descriptor.name: descriptor.display_name for descriptor in descriptors
        }
        self._details = details or ToolProgressFormatter()
        self._text_generation_active = False
        self._sequence = 0

    def thinking(self) -> ConversationProgressEvent:
        return self._event(ProgressKind.THINKING)

    def map(self, event: object) -> tuple[ConversationProgressEvent, ...]:
        """Map one framework event; unsupported and hidden events produce nothing."""
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            mapped: list[ConversationProgressEvent] = []
            if not self._text_generation_active or event.previous_part_kind is None:
                mapped.append(self._event(ProgressKind.TEXT_RESET))
                self._text_generation_active = True
            if event.part.content:
                mapped.append(self._event(ProgressKind.TEXT_DELTA, text=event.part.content))
            return tuple(mapped)
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            if not event.delta.content_delta:
                return ()
            if not self._text_generation_active:
                self._text_generation_active = True
                return (
                    self._event(ProgressKind.TEXT_RESET),
                    self._event(
                        ProgressKind.TEXT_DELTA,
                        text=event.delta.content_delta,
                    ),
                )
            return (
                self._event(
                    ProgressKind.TEXT_DELTA,
                    text=event.delta.content_delta,
                ),
            )
        if isinstance(event, FunctionToolCallEvent):
            self._text_generation_active = False
            part = event.part
            return (
                self._tool_event(
                    ProgressKind.TOOL_STARTED,
                    part.tool_call_id,
                    part.tool_name,
                    self._details.request(part.tool_name, part.args),
                ),
            )
        if isinstance(event, FunctionToolResultEvent):
            part = event.part
            succeeded = (
                isinstance(part, ToolReturnPart)
                and part.outcome == "success"
                and self._model_result_succeeded(part.content)
            )
            return (
                self._tool_event(
                    (ProgressKind.TOOL_SUCCEEDED if succeeded else ProgressKind.TOOL_FAILED),
                    part.tool_call_id,
                    part.tool_name,
                    self._details.result(
                        part.tool_name,
                        part.content,
                        succeeded=succeeded,
                    ),
                ),
            )
        return ()

    @staticmethod
    def _model_result_succeeded(content: object) -> bool:
        if isinstance(content, Mapping):
            return content.get("ok") is True
        return True

    def _tool_event(
        self,
        kind: ProgressKind,
        call_id: str,
        tool_name: str,
        tool_detail: str,
    ) -> ConversationProgressEvent:
        return self._event(
            kind,
            call_id=call_id,
            tool_name=tool_name,
            tool_display_name=self._display_names.get(tool_name, "执行操作"),
            tool_detail=tool_detail,
        )

    def _event(
        self,
        kind: ProgressKind,
        *,
        call_id: str = "",
        tool_name: str = "",
        tool_display_name: str = "",
        tool_detail: str = "",
        text: str = "",
    ) -> ConversationProgressEvent:
        self._sequence += 1
        return ConversationProgressEvent(
            kind=kind,
            event_id=f"engine-{self._sequence}",
            call_id=call_id,
            tool_name=tool_name,
            tool_display_name=tool_display_name,
            tool_detail=tool_detail,
            text=text,
        )
