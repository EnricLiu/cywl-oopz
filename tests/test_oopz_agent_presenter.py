from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest
from oopz_sdk.exceptions import OopzConnectionError

from cywl_oopz.features.chat.models import ChatResponse
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind
from cywl_oopz.integrations.oopz.active_presentations import ActivePresentationRegistry
from cywl_oopz.integrations.oopz.agent_presenter import (
    OopzAgentLoopMessage,
    OopzAgentPresenterFactory,
    OopzPassiveAgentTraceSession,
)
from cywl_oopz.integrations.oopz.editable_messages import (
    EditableMessageRef,
    MessageAddress,
)
from cywl_oopz.integrations.oopz.message_renderer import (
    OopzMessageRenderer,
    oopz_units,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += max(seconds, 0)
        await asyncio.sleep(0)


class FakeGateway:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.edits: list[str] = []
        self.failures: list[Exception | None] = []
        self.active_edits = 0
        self.max_active_edits = 0
        self.edit_started: asyncio.Event | None = None
        self.release_edit: asyncio.Event | None = None
        self.tracked: list[tuple[str, object, object, str]] = []
        self.bound: list[tuple[str, UUID]] = []
        self.finalized: list[tuple[str, dict[str, object]]] = []
        self.superseded: list[tuple[str, dict[str, object]]] = []
        self.promoted: list[tuple[str, UUID | None, dict[str, object]]] = []

    async def create_reply(
        self,
        address: MessageAddress,
        text: str,
    ) -> EditableMessageRef:
        self.created.append(text)
        return EditableMessageRef(
            message_id=f"message-{len(self.created)}",
            timestamp="123",
            scope=address.scope,
            area_id=address.area_id,
            channel_id=address.channel_id,
            target_person_id=address.target_person_id,
            reference_message_id=address.reference_message_id,
        )

    async def replace(self, message: EditableMessageRef, text: str) -> None:
        del message
        self.active_edits += 1
        self.max_active_edits = max(self.max_active_edits, self.active_edits)
        try:
            if self.edit_started is not None:
                self.edit_started.set()
            if self.release_edit is not None:
                await self.release_edit.wait()
            failure = self.failures.pop(0) if self.failures else None
            if failure is not None:
                raise failure
            self.edits.append(text)
        finally:
            self.active_edits -= 1

    async def track_created(
        self,
        message: EditableMessageRef,
        *,
        kind: object,
        state: object,
        owner_person_id: str = "",
    ) -> None:
        self.tracked.append((message.message_id, kind, state, owner_person_id))

    async def bind_agent_run(self, message: EditableMessageRef, run_id: UUID) -> None:
        self.bound.append((message.message_id, run_id))

    async def finalize(
        self,
        message: EditableMessageRef,
        snapshot: dict[str, object],
    ) -> None:
        self.finalized.append((message.message_id, snapshot))

    async def supersede(
        self,
        message: EditableMessageRef,
        snapshot: dict[str, object],
    ) -> None:
        self.superseded.append((message.message_id, snapshot))

    async def promote_agent_response(
        self,
        message: EditableMessageRef,
        run_id: UUID | None,
        snapshot: dict[str, object],
    ) -> None:
        self.promoted.append((message.message_id, run_id, snapshot))


def address() -> MessageAddress:
    return MessageAddress("channel", "area", "channel", "", "source")


async def opened_session(
    gateway: FakeGateway,
) -> OopzAgentLoopMessage:
    clock = FakeClock()
    session = OopzAgentLoopMessage(
        gateway,  # type: ignore[arg-type]
        address(),
        OopzMessageRenderer(),
        edit_interval_seconds=0.8,
        clock=clock,
        sleep=clock.sleep,
    )
    await session.open()
    return session


@pytest.mark.asyncio
async def test_many_text_deltas_coalesce_into_one_serial_terminal_edit() -> None:
    gateway = FakeGateway()
    session = await opened_session(gateway)

    await session.emit(ConversationProgressEvent(ProgressKind.ACCEPTED))
    await session.emit(ConversationProgressEvent(ProgressKind.THINKING))
    await session.emit(ConversationProgressEvent(ProgressKind.TEXT_RESET))
    for _ in range(100):
        await session.emit(ConversationProgressEvent(ProgressKind.TEXT_DELTA, text="字"))
    await session.complete(
        ChatResponse(
            "最终回答",
            "provider/model",
            input_tokens=1200,
            output_tokens=345,
            elapsed_seconds=12.34,
            model_requests=2,
            tool_calls=1,
        )
    )
    await session.aclose()

    assert len(gateway.created) == 1
    assert 1 <= len(gateway.edits) <= 3
    assert gateway.edits[-1] == "🎵 **初音未来** · 12.3s · 1 次工具 · 1.5k tokens\n最终回答"
    assert gateway.max_active_edits == 1
    assert session.state.terminal is True


@pytest.mark.asyncio
async def test_web_research_loop_uses_one_bounded_message_and_safe_stage_names() -> None:
    gateway = FakeGateway()
    session = await opened_session(gateway)

    await session.emit(ConversationProgressEvent(ProgressKind.THINKING))
    await session.emit(
        ConversationProgressEvent(
            ProgressKind.TOOL_STARTED,
            call_id="search-secret-id",
            tool_name="search_web",
            tool_display_name="搜索公开网页",
        )
    )
    await session.emit(
        ConversationProgressEvent(
            ProgressKind.TOOL_SUCCEEDED,
            call_id="search-secret-id",
            tool_name="search_web",
            tool_display_name="搜索公开网页",
        )
    )
    await session.emit(
        ConversationProgressEvent(
            ProgressKind.TOOL_STARTED,
            call_id="read-secret-id",
            tool_name="read_web_page",
            tool_display_name="读取网页正文",
        )
    )
    async with asyncio.timeout(1):
        while not any("读取网页正文" in snapshot for snapshot in gateway.edits):
            await asyncio.sleep(0)

    answer = "已核实的正文。" * 500 + "\n来源：Example 官方文档（https://example.com/docs/current）"
    await session.complete(ChatResponse(answer, "provider/model"))
    await session.aclose()

    snapshots = gateway.created + gateway.edits
    assert len(gateway.created) == 1
    assert any("搜索公开网页" in snapshot for snapshot in gateway.edits)
    assert any("读取网页正文" in snapshot for snapshot in gateway.edits)
    assert all(oopz_units(snapshot) <= 1950 for snapshot in snapshots)
    assert all("secret-id" not in snapshot for snapshot in snapshots)
    assert all("search_web" not in snapshot for snapshot in snapshots)
    assert all("read_web_page" not in snapshot for snapshot in snapshots)
    assert gateway.edits[-1].startswith("🎵 **初音未来**")
    assert gateway.edits[-1].endswith("https://example.com/docs/current）")
    assert session.state.terminal is True


@pytest.mark.asyncio
async def test_skill_share_loop_shows_notification_degradation_in_one_message() -> None:
    gateway = FakeGateway()
    session = await opened_session(gateway)

    await session.emit(
        ConversationProgressEvent(
            ProgressKind.TOOL_STARTED,
            call_id="private-share-id",
            tool_name="invite_agent_skill_share",
            tool_display_name="分享技能",
            tool_subject="当前消息提及的用户",
        )
    )
    await session.emit(
        ConversationProgressEvent(
            ProgressKind.TOOL_SUCCEEDED,
            call_id="private-share-id",
            tool_name="invite_agent_skill_share",
            tool_display_name="分享技能",
            tool_subject="旅行规划",
            tool_summary="已邀请 2 人 · 1 个通知失败",
        )
    )
    async with asyncio.timeout(1):
        while not any("1 个通知失败" in snapshot for snapshot in gateway.edits):
            await asyncio.sleep(0)

    await session.complete(
        ChatResponse(
            "邀请已经保存；一位朋友的私信通知失败，但仍可从技能邀请列表查看。",
            "provider/model",
            elapsed_seconds=1.2,
            model_requests=2,
            tool_calls=1,
        )
    )
    await session.aclose()

    snapshots = gateway.created + gateway.edits
    assert len(gateway.created) == 1
    assert any(
        "✅ **分享技能** 旅行规划 · 已邀请 2 人 · 1 个通知失败" in snapshot
        for snapshot in gateway.edits
    )
    assert "private-share-id" not in repr(snapshots)
    assert "invite_agent_skill_share" not in repr(snapshots)
    assert gateway.edits[-1].startswith("🎵 **初音未来** · 1.2s · 1 次工具")


@pytest.mark.asyncio
async def test_running_tool_heartbeats_without_mutating_reducer_state_and_stops_at_terminal() -> (
    None
):
    gateway = FakeGateway()
    session = OopzAgentLoopMessage(
        gateway,  # type: ignore[arg-type]
        address(),
        OopzMessageRenderer(),
        edit_interval_seconds=0.01,
        heartbeat_interval_seconds=0.12,
    )
    await session.open()
    await session.emit(
        ConversationProgressEvent(
            ProgressKind.TOOL_STARTED,
            call_id="read-call",
            tool_name="read_web_page",
            tool_display_name="读取网页正文",
            tool_subject="www.baidu.com",
        )
    )

    async with asyncio.timeout(1):
        while len(gateway.edits) < 2:
            await asyncio.sleep(0.01)

    assert "· 0.0s" in gateway.edits[0]
    assert "· 0.1s" in gateway.edits[1]
    assert session.state.revision == 1

    await session.complete(ChatResponse("读取完成。", "provider/model"))
    await session.aclose()
    edit_count = len(gateway.edits)
    await asyncio.sleep(0.15)

    assert len(gateway.edits) == edit_count
    assert gateway.edits[-1].startswith("🎵 **初音未来**")


@pytest.mark.asyncio
async def test_provider_retry_is_shown_immediately_and_counted_at_terminal() -> None:
    gateway = FakeGateway()
    session = await opened_session(gateway)

    await session.emit(
        ConversationProgressEvent(
            ProgressKind.MODEL_RETRY,
            retry_attempt=1,
            retry_max_attempts=2,
            retry_delay_seconds=1.25,
            retry_reason="上游服务异常（HTTP 503）",
        )
    )
    async with asyncio.timeout(1):
        while not any("正在重新连接" in snapshot for snapshot in gateway.edits):
            await asyncio.sleep(0)

    retry_snapshot = gateway.edits[-1]
    assert "第 1/2 次重试" in retry_snapshot
    assert "HTTP 503" in retry_snapshot

    await session.emit(ConversationProgressEvent(ProgressKind.THINKING))
    await session.complete(ChatResponse("连接恢复啦♪", "provider/model"))
    await session.aclose()

    assert gateway.edits[-1] == "🎵 **初音未来** · 1 次重试\n连接恢复啦♪"


@pytest.mark.asyncio
async def test_live_response_tracks_run_and_full_terminal_snapshot() -> None:
    gateway = FakeGateway()
    session = await opened_session(gateway)
    run_id = UUID("10000000-0000-0000-0000-000000000001")

    await session.bind_run(run_id)
    await session.emit(
        ConversationProgressEvent(
            ProgressKind.TOOL_STARTED,
            call_id="search-call",
            tool_name="search_web",
            tool_display_name="搜索公开网页",
            tool_subject="初音未来 新闻",
        )
    )
    await session.emit(
        ConversationProgressEvent(
            ProgressKind.TOOL_SUCCEEDED,
            call_id="search-call",
            tool_name="search_web",
            tool_display_name="搜索公开网页",
            tool_subject="初音未来 新闻",
            tool_summary="找到 3 条结果",
        )
    )
    await session.complete(ChatResponse("完整回答", "provider/model", tool_calls=1))
    await session.aclose()

    assert gateway.tracked[0][0] == "message-1"
    assert gateway.bound == [("message-1", run_id)]
    assert gateway.finalized[0][0] == "message-1"
    snapshot = gateway.finalized[0][1]
    assert snapshot["phase"] == "succeeded"
    assert snapshot["final_text"] == "完整回答"
    assert snapshot["tool_calls"] == 1
    assert snapshot["steps"] == [
        {
            "call_id": "search-call",
            "tool_name": "search_web",
            "display_name": "搜索公开网页",
            "status": "succeeded",
            "subject": "初音未来 新闻",
            "summary": "找到 3 条结果",
            "items": [],
            "preview_lines": [],
        }
    ]


@pytest.mark.asyncio
async def test_passive_response_promotes_direct_reply_with_run_snapshot() -> None:
    gateway = FakeGateway()
    session = OopzPassiveAgentTraceSession(gateway, address())  # type: ignore[arg-type]
    run_id = UUID("10000000-0000-0000-0000-000000000001")
    await session.bind_run(run_id)
    await session.emit(
        ConversationProgressEvent(
            ProgressKind.MODEL_RETRY,
            retry_attempt=1,
            retry_max_attempts=2,
            retry_delay_seconds=0.5,
            retry_reason="HTTP 429",
        )
    )

    await session.record_delivery(
        SimpleNamespace(message_id="direct-message", timestamp="456"),
        response=ChatResponse(
            "普通回复的完整答案",
            "provider/model",
            input_tokens=120,
            output_tokens=30,
            model_requests=2,
        ),
    )

    assert gateway.tracked[0][0] == "direct-message"
    message_id, promoted_run, snapshot = gateway.promoted[0]
    assert message_id == "direct-message"
    assert promoted_run == run_id
    assert snapshot["phase"] == "succeeded"
    assert snapshot["final_text"] == "普通回复的完整答案"
    assert snapshot["provider_retry_count"] == 1
    assert snapshot["provider_retries"] == [
        {
            "attempt": 1,
            "max_attempts": 2,
            "delay_seconds": 0.5,
            "reason": "HTTP 429",
        }
    ]


@pytest.mark.asyncio
async def test_dismissed_live_response_never_edits_or_creates_terminal_fallback() -> None:
    gateway = FakeGateway()
    registry = ActivePresentationRegistry()
    session = OopzAgentLoopMessage(
        gateway,  # type: ignore[arg-type]
        address(),
        OopzMessageRenderer(),
        edit_interval_seconds=0.01,
        active_presentations=registry,
    )
    await session.open()

    assert await registry.dismiss("message-1") is True
    await session.emit(ConversationProgressEvent(ProgressKind.THINKING))
    await session.complete(ChatResponse("不应重新出现", "provider/model"))
    await session.aclose()

    assert len(gateway.created) == 1
    assert "不应重新出现" not in gateway.created[0]
    assert gateway.edits == []
    assert gateway.finalized == []
    assert gateway.superseded == []
    assert await registry.dismiss("message-1") is False


@pytest.mark.asyncio
async def test_revision_arriving_during_edit_flushes_the_latest_terminal_snapshot() -> None:
    gateway = FakeGateway()
    gateway.edit_started = asyncio.Event()
    gateway.release_edit = asyncio.Event()
    session = await opened_session(gateway)

    await session.emit(ConversationProgressEvent(ProgressKind.THINKING))
    await asyncio.wait_for(gateway.edit_started.wait(), timeout=1)
    await session.emit(ConversationProgressEvent(ProgressKind.TEXT_RESET))
    await session.emit(ConversationProgressEvent(ProgressKind.TEXT_DELTA, text="新回答"))
    finish = asyncio.create_task(session.complete(ChatResponse("最终新回答", "provider/model")))
    gateway.release_edit.set()
    await asyncio.wait_for(finish, timeout=1)
    await session.aclose()

    assert len(gateway.edits) == 2
    assert gateway.edits[-1].endswith("最终新回答")
    assert gateway.max_active_edits == 1


@pytest.mark.asyncio
async def test_retryable_edit_keeps_latest_state_and_does_not_create_fallback() -> None:
    gateway = FakeGateway()
    gateway.failures = [OopzConnectionError("temporary"), None]
    session = await opened_session(gateway)

    await session.complete(ChatResponse("重试后成功", "provider/model"))
    await session.aclose()

    assert len(gateway.created) == 1
    assert gateway.edits[-1].endswith("重试后成功")


@pytest.mark.asyncio
async def test_deterministic_terminal_edit_failure_creates_exactly_one_fallback() -> None:
    gateway = FakeGateway()
    gateway.failures = [ValueError("unsupported edit")]
    session = await opened_session(gateway)

    await session.complete(ChatResponse("最终回答", "provider/model"))
    await session.aclose()

    assert len(gateway.created) == 2
    assert gateway.created[-1].endswith("最终回答")
    assert gateway.edits == []
    assert gateway.finalized[0][0] == "message-2"
    assert gateway.superseded[0][0] == "message-1"


@pytest.mark.asyncio
async def test_factory_degrades_to_noop_when_placeholder_creation_fails() -> None:
    class BrokenGateway(FakeGateway):
        async def create_reply(
            self,
            address: MessageAddress,
            text: str,
        ) -> EditableMessageRef:
            del address, text
            raise OopzConnectionError("offline")

    factory = OopzAgentPresenterFactory(
        BrokenGateway(),  # type: ignore[arg-type]
        OopzMessageRenderer(),
        enabled=True,
        edit_interval_seconds=0.8,
    )
    message = SimpleNamespace(
        area="area",
        channel="channel",
        sender_id="person",
        message_id="source",
    )
    context = SimpleNamespace(event=SimpleNamespace(message=message, is_private=False))

    session = await factory.open(context)

    assert session.owns_message is False
    await session.complete(ChatResponse("normal fallback", "provider/model"))
    await session.aclose()
