"""Text command parsing for the experimental realtime voice feature."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from oopz_sdk.events.context import EventContext

from cywl_oopz.commands.router import ParsedCommand
from cywl_oopz.core.observability import exception_kind, opaque_ref

from .errors import (
    VoiceBackendBusyError,
    VoiceChannelContextRequiredError,
    VoiceConversationError,
    VoiceFeatureDisabledError,
    VoiceRuntimeUnavailableError,
    VoiceSessionAlreadyActiveError,
    VoiceSessionNotActiveError,
    VoiceSessionOwnershipError,
    VoiceSessionStartCancelledError,
    VoiceSessionStartTimeoutError,
    VoiceUserNotInChannelError,
)
from .models import VoiceSessionStatus, VoiceStartRequest, VoiceTextAddress
from .service import VoiceConversationService

logger = logging.getLogger(__name__)


class VoiceCommandPresenter(Protocol):
    """Render command results at the OOPZ integration boundary."""

    async def started(self, context: EventContext, status: VoiceSessionStatus) -> None: ...

    async def stopped(self, context: EventContext, status: VoiceSessionStatus) -> None: ...

    async def status(self, context: EventContext, status: VoiceSessionStatus) -> None: ...

    async def error(self, context: EventContext, message: str) -> None: ...

    async def usage(self, context: EventContext, prefix: str) -> None: ...


class VoiceCommand:
    """Map ``!voice`` subcommands to the application service."""

    name = "voice"
    description = "启动、停止或查看实验性实时语音对话。"

    def __init__(
        self,
        conversations: VoiceConversationService,
        presenter: VoiceCommandPresenter,
        command_prefix: str,
    ) -> None:
        self._conversations = conversations
        self._presenter = presenter
        self._prefix = command_prefix

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        action = command.arguments[0].casefold() if command.arguments else ""
        if action not in {"start", "stop", "status"} or len(command.arguments) != 1:
            await self._presenter.usage(context, self._prefix)
            return
        try:
            owner = self._owner_person_id(context)
            if action == "start":
                status = await self._conversations.start(self._start_request(context, owner))
                await self._presenter.started(context, status)
            elif action == "stop":
                status = await self._conversations.stop(owner)
                await self._presenter.stopped(context, status)
            else:
                await self._presenter.status(context, await self._conversations.status())
        except VoiceConversationError as exc:
            logger.info(
                "Voice command rejected: action=%s owner=%s error=%s",
                action,
                opaque_ref(self._safe_owner(context)),
                exception_kind(exc),
            )
            await self._presenter.error(context, self._error_message(exc))
        except ValueError as exc:
            logger.info("Voice command context invalid: error=%s", exception_kind(exc))
            await self._presenter.error(context, "无法识别发起人或当前文字频道。")
        except Exception as exc:
            logger.error(
                "Unexpected voice command failure: action=%s error=%s",
                action,
                exception_kind(exc),
                exc_info=True,
            )
            await self._presenter.error(context, "语音会话处理失败，请稍后重试。")

    @staticmethod
    def _owner_person_id(context: Any) -> str:
        owner = VoiceCommand._safe_owner(context)
        if not owner:
            raise ValueError("Voice command sender is required")
        return owner

    @staticmethod
    def _safe_owner(context: Any) -> str:
        message = getattr(getattr(context, "event", None), "message", None)
        return str(getattr(message, "sender_id", "")).strip()

    @staticmethod
    def _start_request(context: Any, owner: str) -> VoiceStartRequest:
        event = getattr(context, "event", None)
        if bool(getattr(event, "is_private", False)):
            raise VoiceChannelContextRequiredError
        message = getattr(event, "message", None)
        area_id = str(getattr(message, "area", "")).strip()
        channel_id = str(getattr(message, "channel", "")).strip()
        if not area_id or not channel_id:
            raise VoiceChannelContextRequiredError
        return VoiceStartRequest(owner, VoiceTextAddress(area_id, channel_id))

    @staticmethod
    def _error_message(error: VoiceConversationError) -> str:
        if isinstance(error, VoiceFeatureDisabledError):
            return "实时语音功能当前未启用。"
        if isinstance(error, VoiceChannelContextRequiredError):
            return "请在服务器文字频道中使用语音命令。"
        if isinstance(error, VoiceUserNotInChannelError):
            return "你需要先加入当前区域的一个语音频道。"
        if isinstance(error, VoiceBackendBusyError):
            return "语音频道正被音乐或另一场对话占用，请稍后再试。"
        if isinstance(error, VoiceSessionAlreadyActiveError):
            return "已经有一场语音会话正在进行。"
        if isinstance(error, VoiceSessionNotActiveError):
            return "当前没有进行中的语音会话。"
        if isinstance(error, VoiceSessionOwnershipError):
            return "只有这场语音会话的发起人可以停止它。"
        if isinstance(error, VoiceRuntimeUnavailableError):
            return "实时语音模型尚未配置完成。"
        if isinstance(error, VoiceSessionStartTimeoutError):
            return "加入语音或连接模型超时，请稍后重试。"
        if isinstance(error, VoiceSessionStartCancelledError):
            return "语音会话已在启动过程中停止。"
        return "语音会话暂时不可用，请稍后重试。"
