"""One bounded, editable OOPZ status message for a realtime voice session."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from oopz_sdk.exceptions import OopzConnectionError, OopzRateLimitError

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.voice.models import VoiceSessionState, VoiceSessionStatus

from .editable_messages import EditableMessageRef, MessageAddress, OopzEditableMessageGateway

logger = logging.getLogger(__name__)

VOICE_STATE_LABELS = {
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


class OopzVoiceStatusRenderer:
    """Render only the compact facts useful during repeated voice interaction."""

    def render(
        self,
        status: VoiceSessionStatus,
        *,
        elapsed_seconds: float | None = None,
    ) -> str:
        elapsed = status.elapsed_seconds if elapsed_seconds is None else elapsed_seconds
        model = status.model_display_name or "实时语音"
        state = VOICE_STATE_LABELS[status.state]
        turns = _metric(status, "voice_responses_drained")
        task_calls = _metric(status, "voice_task_control_calls")
        notifications = _metric(status, "voice_task_notifications_presented")
        reconnects = _metric(status, "voice_provider_reconnects") + _metric(
            status, "voice_media_reconnects"
        )
        if status.active:
            title = f"🎙️ **初音未来语音** · {model} · {state}"
        elif status.error_message:
            title = f"⚠️ **语音** {status.error_message}"
        elif status.state is VoiceSessionState.FAILED:
            title = f"⚠️ **语音会话中断** · {model}"
        else:
            title = f"🎵 **语音会话结束** · {model}"
        facts = [_duration(elapsed), f"{turns} 轮"]
        if task_calls:
            facts.append(f"后台调用 {task_calls}")
        if notifications:
            facts.append(f"通知 {notifications}")
        if reconnects:
            facts.append(f"重连 {reconnects}")
        total_tokens = _usage_total(status)
        if total_tokens:
            facts.append(f"{_compact_number(total_tokens)} tokens")
        return f"{title}\n{' · '.join(facts)}"


class OopzVoiceStatusMessage:
    """Coalesce state snapshots and keep OOPZ message edits off the audio path."""

    max_retryable_failures = 3

    def __init__(
        self,
        gateway: OopzEditableMessageGateway,
        address: MessageAddress,
        *,
        edit_interval_seconds: float = 0.5,
        heartbeat_seconds: float = 1.0,
        renderer: OopzVoiceStatusRenderer | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if edit_interval_seconds <= 0 or heartbeat_seconds <= 0:
            raise ValueError("Voice status display intervals must be positive")
        self._gateway = gateway
        self._address = address
        self._edit_interval = edit_interval_seconds
        self._heartbeat = max(heartbeat_seconds, edit_interval_seconds)
        self._renderer = renderer or OopzVoiceStatusRenderer()
        self._clock = clock
        self._sleep = sleep
        self._message: EditableMessageRef | None = None
        self._status: VoiceSessionStatus | None = None
        self._status_observed_at = 0.0
        self._worker: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._revision = 0
        self._flushed_revision = 0
        self._last_text = ""
        self._next_edit_at = 0.0
        self._retryable_failures = 0
        self._closing = False
        self._disabled = False

    @property
    def owns_message(self) -> bool:
        return self._message is not None

    async def open(self) -> None:
        initial = "🎙️ **初音未来语音** · 正在启动\n00:00 · 0 轮"
        self._message = await self._gateway.create_reply(self._address, initial)
        self._last_text = initial
        self._next_edit_at = self._now() + self._edit_interval
        self._worker = asyncio.create_task(
            self._run_worker(),
            name="oopz-voice-status-message",
        )
        logger.info("Opened OOPZ voice status display: address=%s", self._address_ref())

    def emit(self, status: VoiceSessionStatus) -> None:
        if self._closing:
            return
        self._status = status
        self._status_observed_at = self._now()
        self._revision += 1
        self._wake.set()

    async def aclose(self) -> None:
        if self._closing:
            worker = self._worker
            if worker is not None:
                await worker
            return
        self._closing = True
        self._wake.set()
        worker = self._worker
        if worker is not None:
            await worker
            self._worker = None
        if self._disabled and self._status is not None and not self._status.active:
            try:
                await self._gateway.create_reply(
                    self._address,
                    self._render(self._status),
                )
            except Exception as exc:
                logger.warning(
                    "OOPZ voice terminal fallback failed: address=%s error=%s",
                    self._address_ref(),
                    type(exc).__name__,
                )

    async def _run_worker(self) -> None:
        while not self._disabled:
            heartbeat = False
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._heartbeat)
            except TimeoutError:
                heartbeat = True
            self._wake.clear()
            status = self._status
            if status is None:
                if self._closing:
                    return
                continue
            revision = self._revision
            terminal = not status.active
            if not terminal:
                delay = self._next_edit_at - self._now()
                if delay > 0:
                    await self._sleep(delay)
            text = self._render(status)
            if text == self._last_text and not (heartbeat and status.active):
                self._flushed_revision = max(self._flushed_revision, revision)
                if terminal or (self._closing and self._revision <= self._flushed_revision):
                    return
                continue
            message = self._message
            if message is None:
                self._disabled = True
                return
            try:
                await self._gateway.replace(message, text)
            except (OopzConnectionError, OopzRateLimitError) as exc:
                self._retryable_failures += 1
                logger.warning(
                    "Retryable OOPZ voice status edit failure %s/%s: address=%s error=%s",
                    self._retryable_failures,
                    self.max_retryable_failures,
                    self._address_ref(),
                    type(exc).__name__,
                )
                if self._retryable_failures >= self.max_retryable_failures:
                    self._disabled = True
                    return
                await self._sleep(self._edit_interval)
                self._wake.set()
                continue
            except Exception as exc:
                logger.warning(
                    "OOPZ voice status edits disabled: address=%s error=%s",
                    self._address_ref(),
                    type(exc).__name__,
                )
                self._disabled = True
                return
            self._retryable_failures = 0
            self._last_text = text
            self._flushed_revision = max(self._flushed_revision, revision)
            self._next_edit_at = self._now() + self._edit_interval
            if terminal:
                return
            if self._closing and self._revision <= self._flushed_revision:
                return
            if self._revision > revision:
                self._wake.set()

    def _render(self, status: VoiceSessionStatus) -> str:
        elapsed = status.elapsed_seconds
        if status.active:
            elapsed += max(0.0, self._now() - self._status_observed_at)
        return self._renderer.render(status, elapsed_seconds=elapsed)

    def _address_ref(self) -> str:
        return opaque_ref(
            self._address.scope,
            self._address.area_id,
            self._address.channel_id,
            self._address.target_person_id,
            self._address.reference_message_id,
        )

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return asyncio.get_running_loop().time()


def _metric(status: VoiceSessionStatus, key: str) -> int:
    value = status.metrics.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _usage_total(status: VoiceSessionStatus) -> int:
    value = status.usage.get("total_tokens")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0, int(value))
    total = 0
    for key in ("input_tokens", "output_tokens"):
        item = status.usage.get(key, 0)
        if isinstance(item, int | float) and not isinstance(item, bool):
            total += max(0, int(item))
    return total


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _compact_number(value: int) -> str:
    if value < 1_000:
        return str(value)
    formatted = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
    return f"{formatted}k"
