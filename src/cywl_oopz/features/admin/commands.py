"""Privileged administration commands."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    AccessRequirement,
    CommandDefinition,
    CommandExecutionPolicy,
    CommandUsageError,
    ExecutionMode,
    NoArguments,
    NoArgumentsParser,
)
from cywl_oopz.commands.models import CommandRequest, CommandScope
from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.access.models import AccessResource, Permission

from .initialization import ChannelCatalogError, ChannelInitializationService
from .lifecycle import ApplicationLifecycleCoordinator
from .models import (
    AreaInitializationResult,
    ChannelKey,
    MessageRecallOutcome,
    OopzMessageAddress,
    OopzMessageScope,
    ReferencedMessageCandidate,
)
from .ports import AgentDiagnosticRenderer, AgentDiagnosticRepository
from .recall import (
    BotMessageRecallTransportError,
    MessageRecallService,
    ReferencedBotMessageNotFoundError,
)

logger = logging.getLogger(__name__)


class InitTarget(StrEnum):
    CHANNEL = "channel"
    AREA = "area"


@dataclass(frozen=True, slots=True)
class InitArguments:
    target: InitTarget


class InitArgumentsParser:
    """Parse and validate init scope before authorization runs."""

    def parse(self, request: CommandRequest) -> InitArguments:
        if request.location.scope is not CommandScope.CHANNEL:
            raise CommandUsageError(
                "此命令只能在文字频道中使用。",
                include_usage=False,
            )
        assert request.text is not None
        if len(request.text.tokens) > 1:
            raise CommandUsageError("")
        try:
            target = (
                InitTarget(request.text.tokens[0].casefold())
                if request.text.tokens
                else InitTarget.CHANNEL
            )
        except ValueError as exc:
            raise CommandUsageError("") from exc
        return InitArguments(target)


class InitCommandAuthorization:
    """Resolve init permission from the already-parsed target."""

    def is_available(self, request: CommandRequest) -> bool:
        return request.location.scope is CommandScope.CHANNEL

    def requirement(
        self,
        request: CommandRequest,
        arguments: InitArguments,
    ) -> AccessRequirement:
        resource = (
            AccessResource.area(request.location.area_id)
            if arguments.target is InitTarget.AREA
            else AccessResource.channel(
                request.location.area_id,
                request.location.channel_id,
            )
        )
        return AccessRequirement(Permission.CHANNEL_INITIALIZE, resource)

    def visibility_requirement(self, request: CommandRequest) -> AccessRequirement:
        return AccessRequirement(
            Permission.CHANNEL_INITIALIZE,
            AccessResource.channel(
                request.location.area_id,
                request.location.channel_id,
            ),
        )


class InitCommand:
    """Create missing text/voice settings using database defaults."""

    name = "init"
    description = "初始化当前频道或整个 Area 的 Bot 配置。"
    category = "权限与管理"
    usage = ("init [channel|area]",)

    def __init__(self, service: ChannelInitializationService) -> None:
        self._service = service

    def definition(self) -> CommandDefinition[InitArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            InitArgumentsParser(),
            self,
            InitCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=30.0),
        )

    async def handle(self, request: CommandRequest, arguments: InitArguments) -> None:
        try:
            if arguments.target is InitTarget.AREA:
                result = await self._service.initialize_area(request.location.area_id)
                await request.responder.reply(self._area_result(result))
                return
            result = await self._service.initialize_channel(
                ChannelKey(
                    request.location.area_id,
                    request.location.channel_id,
                )
            )
        except ChannelCatalogError as exc:
            logger.warning(
                "Area initialization discovery unavailable: area=%s error=%s",
                opaque_ref(request.location.area_id),
                type(exc).__name__,
            )
            await request.responder.reply("无法读取 Area 频道列表，请稍后重试。")
            return
        except DatabaseError as exc:
            logger.warning(
                "Channel initialization persistence unavailable: resource=%s error=%s",
                opaque_ref(request.location.area_id, request.location.channel_id),
                type(exc).__name__,
            )
            await request.responder.reply("频道初始化服务暂时不可用，请稍后重试。")
            return

        if result.created:
            await request.responder.reply(
                "✅ **频道已初始化**\n已使用默认配置创建，现有频道设置未改动。"
            )
        else:
            await request.responder.reply("频道已经初始化，现有配置未改动。")

    @staticmethod
    def _area_result(result: AreaInitializationResult) -> str:
        return (
            "✅ **Area 初始化完成**\n"
            f"文字频道：新增 {result.text_created} · 已存在 {result.text_existing}\n"
            f"语音频道：新增 {result.voice_created} · 已存在 {result.voice_existing}\n"
            "现有配置均未改动。"
        )


@dataclass(frozen=True, slots=True)
class DebugArguments:
    reference_message_id: str
    verbose: bool


class DebugArgumentsParser:
    def parse(self, request: CommandRequest) -> DebugArguments:
        assert request.text is not None
        if request.text.tokens not in {(), ("-v",), ("--verbose",)}:
            raise CommandUsageError("")
        if request.target is None:
            raise CommandUsageError(
                "请先引用一条 CYWL Agent 回复。",
                include_usage=False,
            )
        return DebugArguments(
            request.target.message_id,
            bool(request.text.tokens),
        )


class DebugCommandAuthorization:
    def is_available(self, request: CommandRequest) -> bool:
        del request
        return True

    def requirement(
        self,
        request: CommandRequest,
        arguments: DebugArguments,
    ) -> AccessRequirement:
        del arguments
        return AccessRequirement(
            Permission.AGENT_RESPONSE_DEBUG,
            _request_resource(request),
        )

    def visibility_requirement(self, request: CommandRequest) -> AccessRequirement:
        return AccessRequirement(
            Permission.AGENT_RESPONSE_DEBUG,
            _request_resource(request),
        )


class DebugCommand:
    """Expand a referenced tracked Agent response into bounded diagnostic pages."""

    name = "debug"
    description = "展开引用的 Agent 回复及工具调用详情。"
    category = "权限与管理"
    usage = ("debug [-v|--verbose]（请引用 Agent 回复）",)
    timeout_seconds = 10.0

    def __init__(
        self,
        repository: AgentDiagnosticRepository,
        renderer: AgentDiagnosticRenderer,
    ) -> None:
        self._repository = repository
        self._renderer = renderer

    def definition(self) -> CommandDefinition[DebugArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            DebugArgumentsParser(),
            self,
            DebugCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=12.0),
        )

    async def handle(self, request: CommandRequest, arguments: DebugArguments) -> None:
        try:
            address = _request_message_address(request)
            async with asyncio.timeout(self.timeout_seconds):
                diagnostic = await self._repository.get_by_outbound_message(
                    arguments.reference_message_id,
                    address,
                )
        except (DatabaseError, TimeoutError) as exc:
            logger.warning(
                "Agent diagnostic lookup unavailable: message=%s error=%s",
                opaque_ref(arguments.reference_message_id),
                type(exc).__name__,
            )
            await request.responder.reply("诊断服务暂时不可用，请稍后重试。")
            return
        if diagnostic is None:
            await request.responder.reply("引用的消息没有可用的 Agent 运行详情。")
            return
        pages = self._renderer.render(diagnostic, verbose=arguments.verbose)
        for page in pages:
            await request.responder.reply(page)
        logger.info(
            "Agent diagnostic rendered: message=%s pages=%s verbose=%s",
            opaque_ref(arguments.reference_message_id),
            len(pages),
            arguments.verbose,
        )


@dataclass(frozen=True, slots=True)
class RecallArguments:
    reference_message_id: str
    embedded: ReferencedMessageCandidate | None


class RecallArgumentsParser:
    def parse(self, request: CommandRequest) -> RecallArguments:
        assert request.text is not None
        if request.text.tokens:
            raise CommandUsageError("")
        if request.target is None:
            raise CommandUsageError(
                "请先引用一条 CYWL 回复。",
                include_usage=False,
            )
        evidence = request.target.evidence
        return RecallArguments(
            request.target.message_id,
            evidence if isinstance(evidence, ReferencedMessageCandidate) else None,
        )


class RecallCommandAuthorization:
    def is_available(self, request: CommandRequest) -> bool:
        del request
        return True

    def requirement(
        self,
        request: CommandRequest,
        arguments: RecallArguments,
    ) -> AccessRequirement:
        del arguments
        return AccessRequirement(Permission.BOT_MESSAGE_RECALL, _request_resource(request))

    def visibility_requirement(self, request: CommandRequest) -> AccessRequirement:
        return AccessRequirement(Permission.BOT_MESSAGE_RECALL, _request_resource(request))


class RecallCommand:
    """Recall one explicitly quoted Bot-owned channel or private message."""

    name = "recall"
    description = "撤回引用的一条 CYWL 回复。"
    category = "权限与管理"
    usage = ("recall（请引用一条 CYWL 回复）",)
    timeout_seconds = 10.0

    def __init__(
        self,
        service: MessageRecallService,
    ) -> None:
        self._service = service

    def definition(self) -> CommandDefinition[RecallArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            RecallArgumentsParser(),
            self,
            RecallCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=12.0),
        )

    async def handle(self, request: CommandRequest, arguments: RecallArguments) -> None:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                outcome = await self._service.recall(
                    arguments.reference_message_id,
                    _request_message_address(request),
                    arguments.embedded,
                )
        except ReferencedBotMessageNotFoundError:
            await request.responder.reply("引用的消息不是可撤回的 CYWL 回复。")
            return
        except (BotMessageRecallTransportError, TimeoutError) as exc:
            logger.warning(
                "Bot message recall unavailable: message=%s error=%s",
                opaque_ref(arguments.reference_message_id),
                type(exc).__name__,
            )
            await request.responder.reply("撤回失败，请稍后重试。")
            return
        except DatabaseError as exc:
            logger.warning(
                "Bot message recall persistence unavailable: message=%s error=%s",
                opaque_ref(arguments.reference_message_id),
                type(exc).__name__,
            )
            await request.responder.reply("撤回服务暂时不可用，请稍后重试。")
            return
        if outcome is MessageRecallOutcome.ALREADY_RECALLED:
            await request.responder.reply("这条回复已经撤回。")
            return
        await self._confirm_responder(request, arguments.reference_message_id)

    @staticmethod
    async def _confirm_responder(request: CommandRequest, reference_id: str) -> None:
        try:
            result = await request.responder.react("✅")
            if hasattr(result, "ok") and not bool(result.ok):
                raise RuntimeError("OOPZ rejected confirmation reaction")
        except Exception as exc:
            logger.warning(
                "Recall confirmation reaction degraded: message=%s error=%s",
                opaque_ref(reference_id),
                type(exc).__name__,
            )


def _request_resource(request: CommandRequest) -> AccessResource:
    if request.location.scope is CommandScope.PRIVATE:
        return AccessResource.private()
    return AccessResource.channel(
        request.location.area_id,
        request.location.channel_id,
    )


def _request_message_address(request: CommandRequest) -> OopzMessageAddress:
    private = request.location.scope is CommandScope.PRIVATE
    return OopzMessageAddress(
        OopzMessageScope.PRIVATE if private else OopzMessageScope.CHANNEL,
        area_id="" if private else request.location.area_id,
        channel_id=request.location.channel_id,
        target_person_id=request.actor.person_id if private else "",
    )


class RebootCommandAuthorization:
    def is_available(self, request: CommandRequest) -> bool:
        del request
        return True

    def requirement(
        self,
        request: CommandRequest,
        arguments: NoArguments,
    ) -> AccessRequirement:
        del request, arguments
        return AccessRequirement(Permission.BOT_REBOOT, AccessResource.global_resource())

    def visibility_requirement(self, request: CommandRequest) -> AccessRequirement:
        del request
        return AccessRequirement(Permission.BOT_REBOOT, AccessResource.global_resource())


class RebootCommand:
    """Request graceful application exit for an external supervisor to restart."""

    name = "reboot"
    description = "优雅退出并请求外部进程管理器重启 Bot。"
    category = "权限与管理"
    usage = ("reboot",)

    def __init__(self, lifecycle: ApplicationLifecycleCoordinator) -> None:
        self._lifecycle = lifecycle

    def definition(self) -> CommandDefinition[NoArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            NoArgumentsParser(),
            self,
            RebootCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=10.0),
        )

    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del arguments
        actor_ref = opaque_ref(request.actor.person_id)

        async def confirm() -> object:
            return await request.responder.reply("🔄 **正在重启…**")

        accepted = await self._lifecycle.request_restart(actor_ref, confirm)
        if not accepted:
            await request.responder.reply("重启已经在进行中。")
