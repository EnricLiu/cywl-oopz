"""Map Pydantic AI stream events to project-owned display-safe progress."""

from __future__ import annotations

import logging
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

from cywl_oopz.core.observability import exception_kind
from cywl_oopz.features.chat.progress import (
    ConversationProgressEvent,
    ProgressKind,
    ProgressSink,
    emit_progress,
)

from .tool_progress import ToolProgressCatalog, ToolProgressPresentation
from .tools.models import ToolDescriptor, ToolProgressUpdate

logger = logging.getLogger(__name__)


class ConversationToolProgressReporter:
    """Bind safe tool updates to one conversation call identity."""

    def __init__(
        self,
        progress: ProgressSink,
        *,
        call_id: str,
        tool_name: str,
        tool_display_name: str,
    ) -> None:
        self._progress = progress
        self._call_id = call_id
        self._tool_name = tool_name
        self._tool_display_name = tool_display_name
        self._sequence = 0

    async def update(self, update: ToolProgressUpdate) -> None:
        self._sequence += 1
        await emit_progress(
            self._progress,
            ConversationProgressEvent(
                ProgressKind.TOOL_UPDATED,
                event_id=f"tool-update-{self._call_id}-{self._sequence}",
                call_id=self._call_id,
                tool_name=self._tool_name,
                tool_display_name=self._tool_display_name,
                tool_subject=update.subject,
                tool_summary=update.summary,
                tool_items=update.items,
                tool_preview_lines=update.preview_lines,
            ),
        )


class PydanticAiProgressMapper:
    """Stateful mapper for model turns without exposing reasoning or tool data."""

    def __init__(
        self,
        descriptors: tuple[ToolDescriptor, ...],
        details: ToolProgressCatalog | None = None,
    ) -> None:
        self._display_names = {
            descriptor.name: descriptor.display_name for descriptor in descriptors
        }
        self._details = details or ToolProgressCatalog()
        self._text_generation_active = False
        self._sequence = 0

    def thinking(self) -> ConversationProgressEvent:
        return self._event(ProgressKind.THINKING)

    def map(self, event: object) -> tuple[ConversationProgressEvent, ...]:
        """Best-effort map one framework event without controlling Agent success."""
        try:
            return self._map(event)
        except Exception as exc:
            logger.error(
                "Skipped invalid Agent progress projection: event=%s error=%s",
                type(event).__name__,
                exception_kind(exc),
                exc_info=True,
            )
            return ()

    def _map(self, event: object) -> tuple[ConversationProgressEvent, ...]:
        """Map supported framework events after the best-effort boundary."""
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
        presentation: ToolProgressPresentation,
    ) -> ConversationProgressEvent:
        return self._event(
            kind,
            call_id=call_id,
            tool_name=tool_name,
            tool_display_name=self._display_names.get(tool_name, "执行操作"),
            tool_subject=presentation.subject,
            tool_summary=presentation.summary,
            tool_items=presentation.items,
            tool_preview_lines=presentation.preview_lines,
        )

    def _event(
        self,
        kind: ProgressKind,
        *,
        call_id: str = "",
        tool_name: str = "",
        tool_display_name: str = "",
        tool_subject: str = "",
        tool_summary: str = "",
        tool_items: tuple[str, ...] = (),
        tool_preview_lines: tuple[str, ...] = (),
        text: str = "",
    ) -> ConversationProgressEvent:
        self._sequence += 1
        return ConversationProgressEvent(
            kind=kind,
            event_id=f"engine-{self._sequence}",
            call_id=call_id,
            tool_name=tool_name,
            tool_display_name=tool_display_name,
            tool_subject=tool_subject,
            tool_summary=tool_summary,
            tool_items=tool_items,
            tool_preview_lines=tool_preview_lines,
            text=text,
        )
