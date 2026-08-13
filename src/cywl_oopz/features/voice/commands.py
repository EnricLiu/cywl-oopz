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
    VoiceChannelDisabledError,
    VoiceConfigurationUnavailableError,
    VoiceConversationError,
    VoiceFeatureDisabledError,
    VoiceModelSelectionError,
    VoiceRuntimeUnavailableError,
    VoiceSessionAlreadyActiveError,
    VoiceSessionNotActiveError,
    VoiceSessionOwnershipError,
    VoiceSessionStartCancelledError,
    VoiceSessionStartTimeoutError,
    VoiceSpeakerSelectionError,
    VoiceUserNotInChannelError,
)
from .models import VoiceSessionState, VoiceSessionStatus, VoiceStartRequest, VoiceTextAddress
from .ports import VoiceConfigurationRepository, VoiceSessionStatusSink
from .service import VoiceConversationService
from .settings import SelectableVoiceModel, VoiceUserSelection

logger = logging.getLogger(__name__)


class VoiceCommandPresenter(Protocol):
    """Render command results at the OOPZ integration boundary."""

    async def open_session(self, context: EventContext) -> VoiceSessionStatusSink: ...

    async def started(self, context: EventContext, status: VoiceSessionStatus) -> None: ...

    async def stopped(self, context: EventContext, status: VoiceSessionStatus) -> None: ...

    async def status(self, context: EventContext, status: VoiceSessionStatus) -> None: ...

    async def models(
        self,
        context: EventContext,
        models: tuple[SelectableVoiceModel, ...],
    ) -> None: ...

    async def model_selected(self, context: EventContext, model: SelectableVoiceModel) -> None: ...

    async def voice_selected(
        self, context: EventContext, selection: VoiceUserSelection
    ) -> None: ...

    async def error(self, context: EventContext, message: str) -> None: ...

    async def usage(self, context: EventContext, prefix: str) -> None: ...


class VoiceCommand:
    """Map ``!voice`` subcommands to the application service."""

    name = "voice"
    description = "启动、停止或查看实验性实时语音对话。"
    category = "语音"
    usage = (
        "voice <start|stop|status>",
        "voice model [模型别名]",
        "voice voice [音色]",
    )

    def __init__(
        self,
        conversations: VoiceConversationService,
        configurations: VoiceConfigurationRepository,
        presenter: VoiceCommandPresenter,
        command_prefix: str,
    ) -> None:
        self._conversations = conversations
        self._configurations = configurations
        self._presenter = presenter
        self._prefix = command_prefix

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        action = command.arguments[0].casefold() if command.arguments else ""
        valid = (action in {"start", "stop", "status"} and len(command.arguments) == 1) or (
            action in {"model", "voice"} and len(command.arguments) in {1, 2}
        )
        if not valid:
            await self._presenter.usage(context, self._prefix)
            return
        display: VoiceSessionStatusSink | None = None
        try:
            owner = self._owner_person_id(context)
            if action == "start":
                request = self._start_request(context, owner)
                display = await self._presenter.open_session(context)
                status = await self._conversations.start(
                    request,
                    display,
                )
                if not display.owns_message:
                    await self._presenter.started(context, status)
            elif action == "stop":
                status = await self._conversations.stop(owner)
                await self._presenter.stopped(context, status)
            elif action == "status":
                await self._presenter.status(context, await self._conversations.status())
            elif action == "model":
                if len(command.arguments) == 1:
                    await self._presenter.models(
                        context,
                        await self._configurations.list_selectable_models(owner),
                    )
                else:
                    selected = await self._configurations.set_user_model(
                        owner,
                        command.arguments[1],
                    )
                    await self._presenter.model_selected(context, selected)
            elif len(command.arguments) == 1:
                await self._presenter.voice_selected(
                    context,
                    await self._configurations.user_selection(owner),
                )
            else:
                await self._configurations.set_user_voice(owner, command.arguments[1])
                await self._presenter.voice_selected(
                    context,
                    await self._configurations.user_selection(owner),
                )
        except VoiceConversationError as exc:
            logger.info(
                "Voice command rejected: action=%s owner=%s error=%s",
                action,
                opaque_ref(self._safe_owner(context)),
                exception_kind(exc),
            )
            message = self._error_message(exc)
            if not await self._finish_display_error(display, message):
                await self._presenter.error(context, message)
        except ValueError as exc:
            logger.info("Voice command context invalid: error=%s", exception_kind(exc))
            message = "无法识别发起人或当前文字频道。"
            if not await self._finish_display_error(display, message):
                await self._presenter.error(context, message)
        except Exception as exc:
            logger.error(
                "Unexpected voice command failure: action=%s error=%s",
                action,
                exception_kind(exc),
                exc_info=True,
            )
            message = "语音会话处理失败，请稍后重试。"
            if not await self._finish_display_error(display, message):
                await self._presenter.error(context, message)

    @staticmethod
    async def _finish_display_error(
        display: VoiceSessionStatusSink | None,
        message: str,
    ) -> bool:
        if display is None or not display.owns_message:
            return False
        display.emit(
            VoiceSessionStatus(
                active=False,
                state=VoiceSessionState.FAILED,
                error_message=message,
            )
        )
        try:
            await display.aclose()
        except Exception as exc:
            logger.warning("Voice error display finalization failed: error=%s", exception_kind(exc))
        return True

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
        if isinstance(error, VoiceChannelDisabledError):
            return "当前语音频道尚未启用实时对话。"
        if isinstance(error, VoiceConfigurationUnavailableError):
            return "当前没有可用的实时语音模型配置。"
        if isinstance(error, VoiceModelSelectionError):
            return "未找到该语音模型，请使用 provider/model 格式重新选择。"
        if isinstance(error, VoiceSpeakerSelectionError):
            return "音色名称不能为空，且最多 128 个字符。"
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
