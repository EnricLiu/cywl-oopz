"""OOPZ adapters for voice access and compact command presentation."""

from __future__ import annotations

import logging

from oopz_sdk import OopzBot
from oopz_sdk.events.context import EventContext

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.audio.models import (
    AudioChannelKey,
    VoiceParticipantKind,
    VoiceParticipantRequest,
)
from cywl_oopz.features.voice.display import NoopVoiceSessionStatusSink
from cywl_oopz.features.voice.models import VoiceChannelKey, VoiceSessionStatus
from cywl_oopz.features.voice.ports import VoiceSessionStatusSink
from cywl_oopz.features.voice.settings import SelectableVoiceModel, VoiceUserSelection

from .editable_messages import MessageAddress, OopzEditableMessageGateway
from .voice_channel_session import OopzVoiceChannelSessionManager
from .voice_status import VOICE_STATE_LABELS, OopzVoiceStatusMessage

logger = logging.getLogger(__name__)


class OopzConversationVoiceAccess:
    """Map channel lookup and conversation lease requests to the OOPZ SDK."""

    def __init__(self, bot: OopzBot, leases: OopzVoiceChannelSessionManager) -> None:
        self._bot = bot
        self._leases = leases

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        return await self._bot.channels.get_voice_channel_for_user(area_id, person_id)

    async def try_acquire(self, channel: VoiceChannelKey, owner_key: str):
        return await self._leases.try_acquire(
            VoiceParticipantRequest(
                VoiceParticipantKind.CONVERSATION,
                AudioChannelKey(channel.area_id, channel.channel_id),
                owner_key,
            )
        )

    def music_active(self, channel: VoiceChannelKey) -> bool:
        return self._leases.participant_active(
            VoiceParticipantKind.MUSIC,
            AudioChannelKey(channel.area_id, channel.channel_id),
        )


class OopzVoiceCommandPresenter:
    """Open one live status message and render compact command fallbacks."""

    def __init__(
        self,
        editable_messages: OopzEditableMessageGateway | None = None,
        *,
        status_edit_interval_seconds: float = 0.5,
    ) -> None:
        if status_edit_interval_seconds <= 0:
            raise ValueError("Voice status edit interval must be positive")
        self._editable_messages = editable_messages
        self._status_edit_interval = status_edit_interval_seconds
        self._active_display: VoiceSessionStatusSink | None = None

    async def open_session(self, context: EventContext) -> VoiceSessionStatusSink:
        if self._editable_messages is None:
            return NoopVoiceSessionStatusSink()
        try:
            display = OopzVoiceStatusMessage(
                self._editable_messages,
                MessageAddress.from_oopz_context(context),
                edit_interval_seconds=self._status_edit_interval,
            )
            await display.open()
            self._active_display = display
            return display
        except Exception as exc:
            logger.warning(
                "Could not open OOPZ voice status display: error=%s",
                type(exc).__name__,
            )
            return NoopVoiceSessionStatusSink()

    async def started(self, context: EventContext, status: VoiceSessionStatus) -> None:
        await context.reply(f"🎙️ **初音未来语音** · {self._label(status)}")

    async def stopped(self, context: EventContext, status: VoiceSessionStatus) -> None:
        if self._active_display is not None and self._active_display.owns_message:
            self._active_display = None
            return
        await context.reply(f"🎵 **语音会话结束** · {self._duration(status.elapsed_seconds)}")

    async def status(self, context: EventContext, status: VoiceSessionStatus) -> None:
        if not status.active:
            await context.reply("🎙️ **初音未来语音** · 当前空闲")
            return
        channel = "正在定位频道"
        if status.voice_channel is not None:
            channel = (
                f"频道 {opaque_ref(status.voice_channel.area_id, status.voice_channel.channel_id)}"
            )
        mixing = " · 与音乐混流中" if status.music_mixing else ""
        await context.reply(
            f"🎙️ **初音未来语音** · {self._label(status)}\n"
            f"{self._duration(status.elapsed_seconds)} · {channel}{mixing}"
        )

    async def error(self, context: EventContext, message: str) -> None:
        await context.reply(f"⚠️ **语音** {message}")

    async def models(
        self,
        context: EventContext,
        models: tuple[SelectableVoiceModel, ...],
    ) -> None:
        if not models:
            await context.reply("🎙️ **语音模型** · 暂无可选模型")
            return
        lines = ["🎙️ **语音模型**"]
        lines.extend(
            f"{'✅' if model.selected else '▫️'} {model.selector} · {model.display_name}"
            for model in models
        )
        await context.reply("\n".join(lines))

    async def model_selected(self, context: EventContext, model: SelectableVoiceModel) -> None:
        await context.reply(f"✅ **语音模型** {model.selector} · 下次会话生效")

    async def voice_selected(
        self,
        context: EventContext,
        selection: VoiceUserSelection,
    ) -> None:
        voice = selection.voice_id or "模型默认音色"
        await context.reply(f"🎵 **语音音色** {voice} · 下次会话生效")

    async def usage(self, context: EventContext, prefix: str) -> None:
        await context.reply(
            "语音命令：\n"
            f"{prefix}voice start · 开始语音对话\n"
            f"{prefix}voice stop · 结束语音对话\n"
            f"{prefix}voice status · 查看当前状态\n"
            f"{prefix}voice model [provider/model] · 查看或选择模型\n"
            f"{prefix}voice voice [音色] · 查看或选择音色"
        )

    @classmethod
    def _label(cls, status: VoiceSessionStatus) -> str:
        return VOICE_STATE_LABELS[status.state]

    @staticmethod
    def _duration(seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"
