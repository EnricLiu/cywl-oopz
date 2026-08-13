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
from .lifecycle import ApplicationLifecycleCoordinator
from .models import (
    AreaInitializationResult,
    ChannelKey,
    MessageRecallOutcome,
    OopzMessageAddress,
    OopzMessageScope,
)
from .ports import AgentDiagnosticRenderer, AgentDiagnosticRepository, ReferencedMessageParser
from .recall import (
    BotMessageRecallTransportError,
    MessageRecallService,
    ReferencedBotMessageNotFoundError,
)

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
        return _message_address(context)


class RecallCommandAccess:
    """Authorize recall in the exact current message resource."""

    def is_available(self, invocation: OopzAccessInvocation) -> bool:
        del invocation
        return True

    def requirement(
        self,
        command: ParsedCommand,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement:
        del command
        return AccessRequirement(Permission.BOT_MESSAGE_RECALL, invocation.resource)

    def visibility_requirement(
        self,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement:
        return AccessRequirement(Permission.BOT_MESSAGE_RECALL, invocation.resource)


class RecallCommand:
    """Recall one explicitly quoted Bot-owned channel or private message."""

    name = "recall"
    description = "撤回引用的一条 CYWL 回复。"
    timeout_seconds = 10.0

    def __init__(
        self,
        service: MessageRecallService,
        parser: ReferencedMessageParser,
    ) -> None:
        self._service = service
        self._parser = parser

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        if command.arguments:
            await context.reply("用法：/recall（请引用一条 CYWL 回复）")
            return
        message = context.event.message
        reference_id = str(getattr(message, "reference_message_id", "")).strip()
        if not reference_id:
            await context.reply("请先引用一条 CYWL 回复。")
            return
        embedded = self._parser.parse(getattr(message, "reference_message", None))
        try:
            async with asyncio.timeout(self.timeout_seconds):
                outcome = await self._service.recall(
                    reference_id,
                    _message_address(context),
                    embedded,
                )
        except ReferencedBotMessageNotFoundError:
            await context.reply("引用的消息不是可撤回的 CYWL 回复。")
            return
        except (BotMessageRecallTransportError, TimeoutError) as exc:
            logger.warning(
                "Bot message recall unavailable: message=%s error=%s",
                opaque_ref(reference_id),
                type(exc).__name__,
            )
            await context.reply("撤回失败，请稍后重试。")
            return
        except DatabaseError as exc:
            logger.warning(
                "Bot message recall persistence unavailable: message=%s error=%s",
                opaque_ref(reference_id),
                type(exc).__name__,
            )
            await context.reply("撤回服务暂时不可用，请稍后重试。")
            return
        if outcome is MessageRecallOutcome.ALREADY_RECALLED:
            await context.reply("这条回复已经撤回。")
            return
        await self._confirm(context, reference_id)

    @staticmethod
    async def _confirm(context: EventContext, reference_id: str) -> None:
        try:
            result = await context.react("✅")
            if hasattr(result, "ok") and not bool(result.ok):
                raise RuntimeError("OOPZ rejected confirmation reaction")
        except Exception as exc:
            logger.warning(
                "Recall confirmation reaction degraded: message=%s error=%s",
                opaque_ref(reference_id),
                type(exc).__name__,
            )


def _message_address(context: EventContext) -> OopzMessageAddress:
    event = context.event
    message = event.message
    private = bool(getattr(event, "is_private", False))
    return OopzMessageAddress(
        OopzMessageScope.PRIVATE if private else OopzMessageScope.CHANNEL,
        area_id="" if private else str(getattr(message, "area", "")),
        channel_id=str(getattr(message, "channel", "")),
        target_person_id=(str(getattr(message, "sender_id", "")) if private else ""),
    )


class RebootCommandAccess:
    """Require a global reboot permission regardless of invocation channel."""

    def is_available(self, invocation: OopzAccessInvocation) -> bool:
        del invocation
        return True

    def requirement(
        self,
        command: ParsedCommand,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement:
        del command, invocation
        return AccessRequirement(Permission.BOT_REBOOT, AccessResource.global_resource())

    def visibility_requirement(
        self,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement:
        del invocation
        return AccessRequirement(Permission.BOT_REBOOT, AccessResource.global_resource())


class RebootCommand:
    """Request graceful application exit for an external supervisor to restart."""

    name = "reboot"
    description = "优雅退出并请求外部进程管理器重启 Bot。"

    def __init__(self, lifecycle: ApplicationLifecycleCoordinator) -> None:
        self._lifecycle = lifecycle

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        if command.arguments:
            await context.reply("用法：/reboot")
            return
        actor_ref = opaque_ref(str(getattr(context.event.message, "sender_id", "")))

        async def confirm() -> object:
            return await context.reply("🔄 **正在重启…**")

        accepted = await self._lifecycle.request_restart(actor_ref, confirm)
        if not accepted:
            await context.reply("重启已经在进行中。")
