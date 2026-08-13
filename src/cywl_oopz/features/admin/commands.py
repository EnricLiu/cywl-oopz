"""Privileged administration commands."""

from __future__ import annotations

import asyncio
import logging

from oopz_sdk.events.context import EventContext

from cywl_oopz.commands.router import AccessRequirement, ParsedCommand
from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.access.models import AccessResource, AccessResourceKind, Permission
from cywl_oopz.integrations.oopz.access import OopzAccessInvocation

from .initialization import ChannelCatalogError, ChannelInitializationService
from .models import (
    AreaInitializationResult,
    ChannelKey,
    OopzMessageAddress,
    OopzMessageScope,
)
from .ports import AgentDiagnosticRenderer, AgentDiagnosticRepository

logger = logging.getLogger(__name__)


class InitCommandAccess:
    """Authorize channel and area initialization against their exact scope."""

    def is_available(self, invocation: OopzAccessInvocation) -> bool:
        return invocation.resource.kind is AccessResourceKind.CHANNEL

    def requirement(
        self,
        command: ParsedCommand,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement:
        resource = invocation.resource
        if (
            command.arguments
            and command.arguments[0].casefold() == "area"
            and resource.kind is AccessResourceKind.CHANNEL
        ):
            resource = AccessResource.area(resource.area_id)
        return AccessRequirement(Permission.CHANNEL_INITIALIZE, resource)

    def visibility_requirement(
        self,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement:
        return AccessRequirement(Permission.CHANNEL_INITIALIZE, invocation.resource)


class InitCommand:
    """Create missing text/voice settings using database defaults."""

    name = "init"
    description = "初始化当前频道或整个 Area 的 Bot 配置。"

    def __init__(self, service: ChannelInitializationService) -> None:
        self._service = service

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        invocation = OopzAccessInvocation.from_context(context)
        if invocation.resource.kind is not AccessResourceKind.CHANNEL:
            await context.reply("/init 只能在文字频道中使用。")
            return
        if len(command.arguments) > 1 or (
            command.arguments and command.arguments[0].casefold() not in {"channel", "area"}
        ):
            await context.reply("用法：/init [channel|area]")
            return

        target = command.arguments[0].casefold() if command.arguments else "channel"
        try:
            if target == "area":
                result = await self._service.initialize_area(invocation.resource.area_id)
                await context.reply(self._area_result(result))
                return
            result = await self._service.initialize_channel(
                ChannelKey(
                    invocation.resource.area_id,
                    invocation.resource.channel_id,
                )
            )
        except ChannelCatalogError as exc:
            logger.warning(
                "Area initialization discovery unavailable: area=%s error=%s",
                opaque_ref(invocation.resource.area_id),
                type(exc).__name__,
            )
            await context.reply("无法读取 Area 频道列表，请稍后重试。")
            return
        except DatabaseError as exc:
            logger.warning(
                "Channel initialization persistence unavailable: resource=%s error=%s",
                opaque_ref(
                    invocation.resource.area_id,
                    invocation.resource.channel_id,
                ),
                type(exc).__name__,
            )
            await context.reply("频道初始化服务暂时不可用，请稍后重试。")
            return

        if result.created:
            await context.reply("✅ **频道已初始化**\n已使用默认配置创建，现有频道设置未改动。")
        else:
            await context.reply("频道已经初始化，现有配置未改动。")

    @staticmethod
    def _area_result(result: AreaInitializationResult) -> str:
        return (
            "✅ **Area 初始化完成**\n"
            f"文字频道：新增 {result.text_created} · 已存在 {result.text_existing}\n"
            f"语音频道：新增 {result.voice_created} · 已存在 {result.voice_existing}\n"
            "现有配置均未改动。"
        )


class DebugCommandAccess:
    """Authorize diagnostics in the exact current message resource."""

    def is_available(self, invocation: OopzAccessInvocation) -> bool:
        del invocation
        return True

    def requirement(
        self,
        command: ParsedCommand,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement:
        del command
        return AccessRequirement(Permission.AGENT_RESPONSE_DEBUG, invocation.resource)

    def visibility_requirement(
        self,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement:
        return AccessRequirement(Permission.AGENT_RESPONSE_DEBUG, invocation.resource)


class DebugCommand:
    """Expand a referenced tracked Agent response into bounded diagnostic pages."""

    name = "debug"
    description = "展开引用的 Agent 回复及工具调用详情。"
    timeout_seconds = 10.0

    def __init__(
        self,
        repository: AgentDiagnosticRepository,
        renderer: AgentDiagnosticRenderer,
    ) -> None:
        self._repository = repository
        self._renderer = renderer

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        if command.arguments not in {(), ("-v",), ("--verbose",)}:
            await context.reply("用法：/debug [-v|--verbose]（请引用一条 Agent 回复）")
            return
        message = getattr(getattr(context, "event", None), "message", None)
        reference_id = str(getattr(message, "reference_message_id", "")).strip()
        if not reference_id:
            await context.reply("请先引用一条 CYWL Agent 回复。")
            return
        try:
            address = self._address(context)
            async with asyncio.timeout(self.timeout_seconds):
                diagnostic = await self._repository.get_by_outbound_message(
                    reference_id,
                    address,
                )
        except (DatabaseError, TimeoutError) as exc:
            logger.warning(
                "Agent diagnostic lookup unavailable: message=%s error=%s",
                opaque_ref(reference_id),
                type(exc).__name__,
            )
            await context.reply("诊断服务暂时不可用，请稍后重试。")
            return
        if diagnostic is None:
            await context.reply("引用的消息没有可用的 Agent 运行详情。")
            return
        pages = self._renderer.render(
            diagnostic,
            verbose=bool(command.arguments),
        )
        for page in pages:
            await context.reply(page)
        logger.info(
            "Agent diagnostic rendered: message=%s pages=%s verbose=%s",
            opaque_ref(reference_id),
            len(pages),
            bool(command.arguments),
        )

    @staticmethod
    def _address(context: EventContext) -> OopzMessageAddress:
        event = context.event
        message = event.message
        private = bool(getattr(event, "is_private", False))
        return OopzMessageAddress(
            OopzMessageScope.PRIVATE if private else OopzMessageScope.CHANNEL,
            area_id="" if private else str(getattr(message, "area", "")),
            channel_id=str(getattr(message, "channel", "")),
            target_person_id=(str(getattr(message, "sender_id", "")) if private else ""),
        )
