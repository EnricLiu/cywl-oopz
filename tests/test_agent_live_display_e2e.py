"""Opt-in end-to-end Agent live-display checks.

These tests use the ignored ``.env`` PostgreSQL, LLM, and OOPZ configuration.
Every OOPZ message and Agent thread created here is cleaned up.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import replace

import pytest
from oopz_sdk.events.context import EventContext
from oopz_sdk.models import Message, MessageEvent
from sqlalchemy import delete, select

from cywl_oopz.application import BotApplication
from cywl_oopz.commands.router import ParsedCommand
from cywl_oopz.core.errors import ProviderTimeoutError
from cywl_oopz.features.chat.commands import CancelChatCommand, ChatCommand
from cywl_oopz.features.chat.models import ChatResponse, ConversationKey
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor
from cywl_oopz.integrations.oopz.agent_presenter import OopzAgentPresenterFactory
from cywl_oopz.integrations.oopz.editable_messages import (
    EditableMessageRef,
    MessageAddress,
    OopzEditableMessageGateway,
)
from cywl_oopz.integrations.oopz.message_renderer import OopzMessageRenderer
from cywl_oopz.settings import AgentMode, AppSettings
from cywl_oopz.storage.models import ChannelSettingsRecord


def _live_enabled() -> bool:
    return os.environ.get("CYWL_RUN_LIVE_AGENT_DISPLAY_TESTS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


class RecordingEditableGateway:
    """Count protocol operations while delegating every call to real OOPZ."""

    def __init__(self, delegate: OopzEditableMessageGateway) -> None:
        self._delegate = delegate
        self.created: list[EditableMessageRef] = []
        self.edited_texts: list[str] = []

    async def create_reply(
        self,
        address: MessageAddress,
        text: str,
    ) -> EditableMessageRef:
        created = await self._delegate.create_reply(address, text)
        self.created.append(created)
        return created

    async def replace(self, message: EditableMessageRef, text: str) -> None:
        await self._delegate.replace(message, text)
        self.edited_texts.append(text)


class LiveAgentDisplayHarness:
    """Own real resources and ensure test artifacts are recoverably removed."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = replace(
            settings,
            agent=replace(
                settings.agent,
                mode=AgentMode.AGENT,
                live_display=True,
                display_edit_interval_seconds=0.8,
            ),
        )
        self.application = BotApplication(self.settings)
        self.gateway = RecordingEditableGateway(OopzEditableMessageGateway(self.application.bot))
        self.presenters = OopzAgentPresenterFactory(
            self.gateway,  # type: ignore[arg-type]
            OopzMessageRenderer(),
            enabled=True,
            edit_interval_seconds=0.8,
        )
        self.source: EditableMessageRef | None = None
        self.context: EventContext | None = None
        self.key: ConversationKey | None = None
        self._created_channel_settings = False
        self._original_enabled_tools: list[str] | None = None
        self._channel: tuple[str, str] | None = None

    @classmethod
    async def create(cls) -> LiveAgentDisplayHarness:
        return cls(await AppSettings.from_environment_async())

    async def start(self) -> None:
        await self.application.database.start()
        await self.application.agent_models.reload()
        area_id, channel_id = await self._find_writable_channel()
        self._channel = (area_id, channel_id)
        await self._enable_status_tool(area_id, channel_id)
        sent = await self.application.bot.messages.send_message(
            "🧪 CYWL Agent 单消息端到端验证源消息",
            area=area_id,
            channel=channel_id,
        )
        self.source = EditableMessageRef(
            message_id=sent.message_id,
            timestamp=sent.timestamp,
            scope="channel",
            area_id=area_id,
            channel_id=channel_id,
            target_person_id="",
            reference_message_id="",
        )
        event = MessageEvent(
            event_name="message",
            event_type=1,
            message=Message(
                area=area_id,
                channel=channel_id,
                messageId=sent.message_id,
                timestamp=sent.timestamp,
                person=self.settings.oopz.person_uid,
                text="live display test",
                content="live display test",
            ),
            is_private=False,
        )
        self.context = EventContext(
            bot=self.application.bot,
            config=self.settings.oopz,
            event=event,
        )
        self.key = ConversationKey.from_oopz_context(self.context)

    async def run(self, prompt: str) -> None:
        await self.run_with(self.application.agent_chat, prompt)

    async def run_with(self, service: object, prompt: str) -> None:
        assert self.context is not None
        await ChatCommand(
            service,  # type: ignore[arg-type]
            self.presenters,
        ).execute(
            ParsedCommand("chat", tuple(prompt.split())),
            self.context,
        )

    async def displayed_messages(self) -> list[Message]:
        assert self.source is not None
        messages = await self.application.bot.messages.get_channel_messages(
            self.source.area_id,
            self.source.channel_id,
            size=50,
        )
        return [
            message
            for message in messages
            if message.reference_message_id == self.source.message_id
        ]

    async def persisted_tool_names(self) -> set[str]:
        assert self.key is not None
        thread = await self.application.agent_threads.get(self.key)
        assert thread is not None
        messages = await self.application.agent_messages.load(thread.id, limit=50)
        return {
            str(message.content.get("tool_name"))
            for message in messages
            if message.kind == "tool_call"
        }

    async def aclose(self) -> None:
        await self.application.chat_tasks.close()
        await self.application.agent_summary_tasks.close()
        if self.key is not None:
            with suppress(Exception):
                await self.application.agent_chat.clear(self.key)
        if self._channel is not None:
            with suppress(Exception):
                await self._restore_channel_settings(*self._channel)
        for message in reversed(self.gateway.created):
            with suppress(Exception):
                await self.application.bot.messages.recall_message(
                    message.message_id,
                    area=message.area_id,
                    channel=message.channel_id,
                    timestamp=message.timestamp,
                )
        if self.source is not None:
            with suppress(Exception):
                await self.application.bot.messages.recall_message(
                    self.source.message_id,
                    area=self.source.area_id,
                    channel=self.source.channel_id,
                    timestamp=self.source.timestamp,
                )
        if self.application.music is not None:
            await self.application.music.aclose()
        await self.application.agent_engine.aclose()
        await self.application._provider.aclose()
        await self.application.bot.rest.close()
        await self.application.database.close()

    async def _find_writable_channel(self) -> tuple[str, str]:
        areas = await self.application.bot.areas.get_joined_areas()
        for area in areas:
            groups = await self.application.bot.areas.get_area_channels(area.area_id)
            for channel in (
                channel
                for group in groups
                for channel in group.channels
                if channel.channel_type.casefold() == "text"
            ):
                try:
                    probe = await self.application.bot.messages.send_message(
                        "🧪 CYWL Agent 测试频道探测",
                        area=area.area_id,
                        channel=channel.channel_id,
                    )
                except Exception:
                    continue
                await self.application.bot.messages.recall_message(
                    probe.message_id,
                    area=area.area_id,
                    channel=channel.channel_id,
                    timestamp=probe.timestamp,
                )
                return area.area_id, channel.channel_id
        pytest.skip("no joined OOPZ text channel accepted a temporary bot message")

    async def _enable_status_tool(self, area_id: str, channel_id: str) -> None:
        async with self.application.database.session() as session:
            record = await session.scalar(
                select(ChannelSettingsRecord).where(
                    ChannelSettingsRecord.area_id == area_id,
                    ChannelSettingsRecord.channel_id == channel_id,
                )
            )
            if record is None:
                session.add(
                    ChannelSettingsRecord(
                        area_id=area_id,
                        channel_id=channel_id,
                        chat_enabled=False,
                        enabled_agent_tools=[
                            "get_agent_status",
                            "get_channel_settings",
                        ],
                    )
                )
                self._created_channel_settings = True
                return
            self._original_enabled_tools = list(record.enabled_agent_tools)
            record.enabled_agent_tools = sorted(
                {
                    *record.enabled_agent_tools,
                    "get_agent_status",
                    "get_channel_settings",
                }
            )

    async def _restore_channel_settings(
        self,
        area_id: str,
        channel_id: str,
    ) -> None:
        async with self.application.database.session() as session:
            if self._created_channel_settings:
                await session.execute(
                    delete(ChannelSettingsRecord).where(
                        ChannelSettingsRecord.area_id == area_id,
                        ChannelSettingsRecord.channel_id == channel_id,
                    )
                )
                return
            if self._original_enabled_tools is None:
                return
            record = await session.scalar(
                select(ChannelSettingsRecord).where(
                    ChannelSettingsRecord.area_id == area_id,
                    ChannelSettingsRecord.channel_id == channel_id,
                )
            )
            if record is not None:
                record.enabled_agent_tools = self._original_enabled_tools


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "expected_tools"),
    [
        (
            "不要调用任何工具，只用一句简短中文回复：端到端文字验证完成。",
            (),
        ),
        (
            "必须调用 get_agent_status 一次，拿到结果后用一句简短中文确认完成。",
            ("get_agent_status",),
        ),
        (
            "请并行调用 get_agent_status 和 get_channel_settings 各一次，"
            "拿到两个结果后用一句简短中文确认完成。",
            ("get_agent_status", "get_channel_settings"),
        ),
    ],
)
async def test_live_agent_run_uses_exactly_one_edited_oopz_message(
    prompt: str,
    expected_tools: tuple[str, ...],
) -> None:
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_AGENT_DISPLAY_TESTS=1 to run the live display E2E test")
    harness = await LiveAgentDisplayHarness.create()
    try:
        await harness.start()
        await harness.run(prompt)

        displayed = await harness.displayed_messages()
        assert len(harness.gateway.created) == 1
        assert len(displayed) == 1
        assert harness.gateway.edited_texts
        assert (displayed[0].text or displayed[0].content).startswith("🎵 **CYWL**")
        assert displayed[0].edit_time > 0
        tool_names = await harness.persisted_tool_names()
        if not expected_tools:
            assert tool_names == set()
        else:
            assert set(expected_tools).issubset(tool_names)
    finally:
        await harness.aclose()


@pytest.mark.asyncio
async def test_live_provider_timeout_replaces_the_original_display_message() -> None:
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_AGENT_DISPLAY_TESTS=1 to run the live display E2E test")

    class TimeoutService:
        enabled = True

        async def ask(self, *args: object, **kwargs: object) -> ChatResponse:
            del args
            progress = kwargs["progress"]
            await progress.emit(ConversationProgressEvent(ProgressKind.ACCEPTED))
            await progress.emit(ConversationProgressEvent(ProgressKind.THINKING))
            raise ProviderTimeoutError("simulated timeout")

    harness = await LiveAgentDisplayHarness.create()
    try:
        await harness.start()
        await harness.run_with(TimeoutService(), "触发超时")

        displayed = await harness.displayed_messages()
        assert len(harness.gateway.created) == 1
        assert len(displayed) == 1
        assert "模型响应超时" in (displayed[0].text or displayed[0].content)
    finally:
        await harness.aclose()


@pytest.mark.asyncio
async def test_live_tool_failure_is_visible_then_recovers_in_the_same_message() -> None:
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_AGENT_DISPLAY_TESTS=1 to run the live display E2E test")

    class RecoveringService:
        enabled = True

        async def ask(self, *args: object, **kwargs: object) -> ChatResponse:
            del args
            progress = kwargs["progress"]
            await progress.emit(ConversationProgressEvent(ProgressKind.ACCEPTED))
            await progress.emit(ConversationProgressEvent(ProgressKind.THINKING))
            await progress.emit(
                ConversationProgressEvent(
                    ProgressKind.TOOL_STARTED,
                    call_id="simulated-call",
                    tool_name="simulated_lookup",
                    tool_display_name="模拟查询",
                )
            )
            await asyncio.sleep(0.9)
            await progress.emit(
                ConversationProgressEvent(
                    ProgressKind.TOOL_FAILED,
                    call_id="simulated-call",
                    tool_name="simulated_lookup",
                    tool_display_name="模拟查询",
                )
            )
            async with asyncio.timeout(5):
                while not any(
                    "模拟查询未完成" in snapshot for snapshot in harness.gateway.edited_texts
                ):
                    await asyncio.sleep(0.1)
            return ChatResponse("工具失败后已安全恢复。", "simulated/model")

    harness = await LiveAgentDisplayHarness.create()
    try:
        await harness.start()
        await harness.run_with(RecoveringService(), "模拟工具失败")

        displayed = await harness.displayed_messages()
        assert len(harness.gateway.created) == 1
        assert len(displayed) == 1
        assert any("模拟查询未完成" in snapshot for snapshot in harness.gateway.edited_texts)
        assert "工具失败后已安全恢复" in (displayed[0].text or displayed[0].content)
    finally:
        await harness.aclose()


@pytest.mark.asyncio
async def test_live_cancel_replaces_the_original_without_a_second_reply() -> None:
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_AGENT_DISPLAY_TESTS=1 to run the live display E2E test")
    started = asyncio.Event()

    class WaitingService:
        enabled = True

        async def ask(self, *args: object, **kwargs: object) -> ChatResponse:
            del args
            progress = kwargs["progress"]
            await progress.emit(ConversationProgressEvent(ProgressKind.ACCEPTED))
            await progress.emit(ConversationProgressEvent(ProgressKind.THINKING))
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    harness = await LiveAgentDisplayHarness.create()
    supervisor = ChatTaskSupervisor()
    try:
        await harness.start()
        assert harness.context is not None
        assert harness.key is not None
        operation = ChatCommand(
            WaitingService(),
            harness.presenters,
        ).execute(
            ParsedCommand("chat", ("等待",)),
            harness.context,
        )
        assert supervisor.start(harness.key, operation) is True
        await asyncio.wait_for(started.wait(), timeout=2)

        await CancelChatCommand(
            WaitingService(),
            supervisor,
            active_message_reports_cancel=True,
        ).execute(ParsedCommand("cancel", ()), harness.context)

        displayed = await harness.displayed_messages()
        assert len(harness.gateway.created) == 1
        assert len(displayed) == 1
        assert "已取消当前回答" in (displayed[0].text or displayed[0].content)
    finally:
        await supervisor.close()
        await harness.aclose()
