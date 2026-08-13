"""Shared administration actions independent from text or reaction triggers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import exception_kind, opaque_ref

from .models import MessageRecallOutcome, OopzMessageAddress, ReferencedMessageCandidate
from .ports import AgentDiagnosticRenderer, AgentDiagnosticRepository
from .recall import (
    BotMessageRecallTransportError,
    MessageRecallService,
    ReferencedBotMessageNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MessageActionTarget:
    """One exact message selected by any supported administration trigger."""

    message_id: str
    address: OopzMessageAddress
    embedded: ReferencedMessageCandidate | None = None

    def __post_init__(self) -> None:
        message_id = self.message_id.strip()
        if not message_id or len(message_id) > 256:
            raise ValueError("Message action target ID must contain at most 256 characters")
        object.__setattr__(self, "message_id", message_id)


class RecallActionStatus(StrEnum):
    RECALLED = "recalled"
    ALREADY_RECALLED = "already_recalled"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    PERSISTENCE_UNAVAILABLE = "persistence_unavailable"


class RecallMessageAction:
    """Recall a Bot-owned message and return trigger-neutral business outcomes."""

    def __init__(
        self,
        service: MessageRecallService,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Recall action timeout must be positive")
        self._service = service
        self._timeout_seconds = timeout_seconds

    async def execute(self, target: MessageActionTarget) -> RecallActionStatus:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                outcome = await self._service.recall(
                    target.message_id,
                    target.address,
                    target.embedded,
                )
        except ReferencedBotMessageNotFoundError:
            logger.debug(
                "Recall action target is not Bot-owned: message=%s",
                opaque_ref(target.message_id),
            )
            return RecallActionStatus.NOT_APPLICABLE
        except (BotMessageRecallTransportError, TimeoutError) as exc:
            logger.warning(
                "Recall action transport unavailable: message=%s error=%s",
                opaque_ref(target.message_id),
                exception_kind(exc),
            )
            return RecallActionStatus.UNAVAILABLE
        except DatabaseError as exc:
            logger.warning(
                "Recall action persistence unavailable: message=%s error=%s",
                opaque_ref(target.message_id),
                exception_kind(exc),
            )
            return RecallActionStatus.PERSISTENCE_UNAVAILABLE
        if outcome is MessageRecallOutcome.ALREADY_RECALLED:
            return RecallActionStatus.ALREADY_RECALLED
        return RecallActionStatus.RECALLED


class DebugActionStatus(StrEnum):
    COMPLETED = "completed"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class DebugMessageResult:
    status: DebugActionStatus
    pages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status is DebugActionStatus.COMPLETED and not self.pages:
            raise ValueError("Completed debug actions require rendered pages")
        if self.status is not DebugActionStatus.COMPLETED and self.pages:
            raise ValueError("Incomplete debug actions must not contain pages")


class DebugMessageAction:
    """Load and render exact Agent diagnostics for any interaction trigger."""

    def __init__(
        self,
        repository: AgentDiagnosticRepository,
        renderer: AgentDiagnosticRenderer,
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Debug action timeout must be positive")
        self._repository = repository
        self._renderer = renderer
        self._timeout_seconds = timeout_seconds

    async def execute(
        self,
        target: MessageActionTarget,
        *,
        verbose: bool,
    ) -> DebugMessageResult:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                diagnostic = await self._repository.get_by_outbound_message(
                    target.message_id,
                    target.address,
                )
        except (DatabaseError, TimeoutError) as exc:
            logger.warning(
                "Debug action lookup unavailable: message=%s error=%s",
                opaque_ref(target.message_id),
                exception_kind(exc),
            )
            return DebugMessageResult(DebugActionStatus.UNAVAILABLE)
        if diagnostic is None:
            logger.debug(
                "Debug action has no Agent diagnostic: message=%s",
                opaque_ref(target.message_id),
            )
            return DebugMessageResult(DebugActionStatus.NOT_APPLICABLE)
        return DebugMessageResult(
            DebugActionStatus.COMPLETED,
            self._renderer.render(diagnostic, verbose=verbose),
        )
