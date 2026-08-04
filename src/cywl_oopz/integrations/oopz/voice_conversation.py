"""OOPZ adapters for voice access and compact command presentation."""

from __future__ import annotations

from oopz_sdk import OopzBot
from oopz_sdk.events.context import EventContext

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.voice.models import VoiceChannelKey, VoiceSessionState, VoiceSessionStatus
from cywl_oopz.features.voice.settings import SelectableVoiceModel, VoiceUserSelection

from .voice_lease import (
    OopzVoiceLeaseManager,
    VoiceLeasePurpose,
    VoiceLeaseRequest,
)


class OopzConversationVoiceAccess:
    """Map channel lookup and conversation lease requests to the OOPZ SDK."""

    def __init__(self, bot: OopzBot, leases: OopzVoiceLeaseManager) -> None:
        self._bot = bot
        self._leases = leases

    async def voice_channel_for_user(self, area_id: str, person_id: str) -> str | None:
        return await self._bot.channels.get_voice_channel_for_user(area_id, person_id)

    async def try_acquire(self, channel: VoiceChannelKey, owner_key: str):
        return await self._leases.try_acquire(
            VoiceLeaseRequest(
                VoiceLeasePurpose.CONVERSATION,
                channel.area_id,
                channel.channel_id,
                owner_key,
            )
        )


class OopzVoiceCommandPresenter:
    """Render an I2 command skeleton without exposing internal identifiers."""

    _STATE_LABELS = {
        VoiceSessionState.STARTING: "正在启动",
        VoiceSessionState.ACQUIRING_VOICE: "正在加入语音",
        VoiceSessionState.RESOLVING_SPEAKER: "正在定位音轨",
        VoiceSessionState.CONNECTING_PROVIDER: "正在连接模型",
        VoiceSessionState.LISTENING: "正在听",
        VoiceSessionState.USER_SPEAKING: "你在说",
        VoiceSessionState.THINKING: "思考中",
        VoiceSessionState.SPEAKING: "说话中",
        VoiceSessionState.INTERRUPTING: "正在打断",
        VoiceSessionState.RECOVERING: "恢复中",
        VoiceSessionState.CLOSING: "正在结束",
        VoiceSessionState.CLOSED: "已结束",
        VoiceSessionState.FAILED: "已中断",
    }

    async def started(self, context: EventContext, status: VoiceSessionStatus) -> None:
        await context.reply(f"🎙️ **初音未来语音** · {self._label(status)}")

    async def stopped(self, context: EventContext, status: VoiceSessionStatus) -> None:
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
        await context.reply(
            f"🎙️ **初音未来语音** · {self._label(status)}\n"
            f"{self._duration(status.elapsed_seconds)} · {channel}"
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
        return cls._STATE_LABELS[status.state]

    @staticmethod
    def _duration(seconds: float) -> str:
        total = max(0, int(seconds))
        return f"{total // 60:02d}:{total % 60:02d}"
