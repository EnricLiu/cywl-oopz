"""Text command parsing for the experimental realtime voice feature."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    CommandDefinition,
    CommandExecutionPolicy,
    CommandUsageError,
    ExecutionMode,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.models import CommandRequest, CommandScope
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

    async def open_session(self, request: CommandRequest) -> VoiceSessionStatusSink: ...

    async def started(self, request: CommandRequest, status: VoiceSessionStatus) -> None: ...

    async def stopped(self, request: CommandRequest, status: VoiceSessionStatus) -> None: ...

    async def status(self, request: CommandRequest, status: VoiceSessionStatus) -> None: ...

    async def models(
        self,
        request: CommandRequest,
        models: tuple[SelectableVoiceModel, ...],
    ) -> None: ...

    async def model_selected(
        self, request: CommandRequest, model: SelectableVoiceModel
    ) -> None: ...

    async def voice_selected(
        self, request: CommandRequest, selection: VoiceUserSelection
    ) -> None: ...

    async def error(self, request: CommandRequest, message: str) -> None: ...


class VoiceAction(StrEnum):
    START = "start"
    STOP = "stop"
    STATUS = "status"
    MODEL = "model"
    VOICE = "voice"


@dataclass(frozen=True, slots=True)
class VoiceArguments:
    action: VoiceAction
    value: str = ""


class VoiceArgumentsParser:
    def parse(self, request: CommandRequest) -> VoiceArguments:
        assert request.text is not None
        tokens = request.text.tokens
        if not tokens or len(tokens) > 2:
            raise CommandUsageError("请选择一个语音操作。")
        try:
            action = VoiceAction(tokens[0].casefold())
        except ValueError as exc:
            raise CommandUsageError("未知的语音操作。") from exc
        if action in {VoiceAction.START, VoiceAction.STOP, VoiceAction.STATUS}:
            if len(tokens) != 1:
                raise CommandUsageError("此语音操作不接受额外参数。")
            return VoiceArguments(action)
        return VoiceArguments(action, tokens[1] if len(tokens) == 2 else "")


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
    ) -> None:
        self._conversations = conversations
        self._configurations = configurations
        self._presenter = presenter

    def definition(self) -> CommandDefinition[VoiceArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            VoiceArgumentsParser(),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=45.0),
        )

    async def handle(self, request: CommandRequest, arguments: VoiceArguments) -> None:
        action = arguments.action
        display: VoiceSessionStatusSink | None = None
        try:
            owner = request.actor.person_id
            if action is VoiceAction.START:
                start_request = self._start_request(request, owner)
                display = await self._presenter.open_session(request)
                status = await self._conversations.start(
                    start_request,
                    display,
                )
                if not display.owns_message:
                    await self._presenter.started(request, status)
            elif action is VoiceAction.STOP:
                status = await self._conversations.stop(owner)
                await self._presenter.stopped(request, status)
            elif action is VoiceAction.STATUS:
                await self._presenter.status(request, await self._conversations.status())
            elif action is VoiceAction.MODEL:
                if not arguments.value:
                    await self._presenter.models(
                        request,
                        await self._configurations.list_selectable_models(owner),
                    )
                else:
                    selected = await self._configurations.set_user_model(
                        owner,
                        arguments.value,
                    )
                    await self._presenter.model_selected(request, selected)
            elif not arguments.value:
                await self._presenter.voice_selected(
                    request,
                    await self._configurations.user_selection(owner),
                )
            else:
                await self._configurations.set_user_voice(owner, arguments.value)
                await self._presenter.voice_selected(
                    request,
                    await self._configurations.user_selection(owner),
                )
        except VoiceConversationError as exc:
            logger.info(
                "Voice command rejected: action=%s owner=%s error=%s",
                action.value,
                opaque_ref(owner),
                exception_kind(exc),
            )
            message = self._error_message(exc)
            if not await self._finish_display_error(display, message):
                await self._presenter.error(request, message)
        except Exception as exc:
            logger.error(
                "Unexpected voice command failure: action=%s error=%s",
                action.value,
                exception_kind(exc),
                exc_info=True,
            )
            message = "语音会话处理失败，请稍后重试。"
            if not await self._finish_display_error(display, message):
                await self._presenter.error(request, message)

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
    def _start_request(request: CommandRequest, owner: str) -> VoiceStartRequest:
        if request.location.scope is CommandScope.PRIVATE:
            raise VoiceChannelContextRequiredError
        return VoiceStartRequest(
            owner,
            VoiceTextAddress(request.location.area_id, request.location.channel_id),
        )

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
