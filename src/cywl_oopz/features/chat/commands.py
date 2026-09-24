"""OOPZ command controllers for the text-chat use cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    CommandDefinition,
    CommandExecutionPolicy,
    CommandUsageError,
    ExecutionMode,
    NoArguments,
    NoArgumentsParser,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.models import CommandRequest, CommandScope
from cywl_oopz.conversation.input import UserInput as AgentUserInput
from cywl_oopz.conversation.models import ChatInvocation, ChatInvocationFactory, ConversationKey
from cywl_oopz.conversation.progress import ConversationPresenterFactory, NoopPresenterFactory
from cywl_oopz.conversation.responder import ConversationResponder
from cywl_oopz.conversation.use_case import ChatUseCase
from cywl_oopz.core.observability import opaque_ref

from .tasks import ChatTaskSupervisor


class ChatCommandController:
    """Shared safe error mapping for chat-facing command controllers."""

    def __init__(
        self,
        service: ChatUseCase,
        presenter_factory: ConversationPresenterFactory | None = None,
        invocation_factory: ChatInvocationFactory | None = None,
    ) -> None:
        self._service = service
        self._invocations = invocation_factory
        self._responder = ConversationResponder(
            service, presenter_factory or NoopPresenterFactory()
        )

    @staticmethod
    def _request_key(request: CommandRequest) -> ConversationKey:
        private = request.location.scope is CommandScope.PRIVATE
        return ConversationKey(
            scope="private" if private else "channel",
            area_id="" if private else request.location.area_id,
            channel_id="" if private else request.location.channel_id,
            person_id=request.actor.person_id,
        )

    def _request_invocation(self, request: CommandRequest) -> ChatInvocation:
        if self._invocations is not None:
            return self._invocations.from_request(request)
        excluded = {request.actor.person_id}
        return ChatInvocation(
            source_message_id=request.source.message_id,
            transport_channel_id=request.location.channel_id,
            mentioned_person_ids=tuple(
                mention.person_id
                for mention in request.mentions
                if mention.person_id not in excluded
            ),
        )

    @staticmethod
    def _request_ref(request: CommandRequest, key: ConversationKey) -> str:
        return opaque_ref(
            "chat-command",
            request.source.message_id,
            key.scope,
            key.area_id,
            key.channel_id,
            key.person_id,
        )

    async def _reply_request_error(
        self,
        request: CommandRequest,
        error: Exception,
    ) -> None:
        key = self._request_key(request)
        await self._responder.reply_error(
            error,
            request_ref=self._request_ref(request, key),
            reply=request.responder.reply,
            conversation_ref=opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id),
        )

    async def _ask_request_with_presenter(
        self,
        request: CommandRequest,
        prompt: str,
        *,
        user_input: AgentUserInput | None = None,
    ) -> bool:
        """Run the command path using only project-owned request values."""
        key = self._request_key(request)
        return await self._responder.run(
            presenter_context=request,
            key=key,
            prompt=prompt,
            user_input=user_input,
            invocation=self._request_invocation(request),
            request_ref=self._request_ref(request, key),
            reply=request.responder.reply,
        )


@dataclass(frozen=True, slots=True)
class ChatArguments:
    prompt: str


class ChatArgumentsParser:
    def parse(self, request: CommandRequest) -> ChatArguments:
        assert request.text is not None
        prompt = request.text.raw_tail.strip()
        if not prompt:
            raise CommandUsageError("请在命令后附上想说的内容。")
        return ChatArguments(prompt)


class ChatCommand(ChatCommandController):
    """Start or continue a text conversation with `/chat <prompt>`."""

    name = "chat"
    description = "向 LLM 发起或继续文字对话。"
    category = "对话"
    usage = ("chat <内容>",)

    def __init__(
        self,
        service: ChatUseCase,
        tasks: ChatTaskSupervisor,
        presenter_factory: ConversationPresenterFactory | None = None,
        invocation_factory: ChatInvocationFactory | None = None,
        *,
        prefix: str = "/",
    ) -> None:
        super().__init__(service, presenter_factory, invocation_factory)
        self._tasks = tasks
        self._prefix = prefix

    def definition(self) -> CommandDefinition[ChatArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            ChatArgumentsParser(),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND),
        )

    async def handle(self, request: CommandRequest, arguments: ChatArguments) -> None:
        key = self._request_key(request)
        if not self._tasks.start(
            key,
            self._ask_request_with_presenter(
                request,
                arguments.prompt,
                user_input=request.user_input,
            ),
        ):
            await request.responder.reply(
                f"当前对话正在生成回复；可使用 {self._prefix}cancel 取消后再试。"
            )
            return
        await self._tasks.wait(key)


class NewConversationCommand(ChatCommandController):
    """Forget only the caller's active conversation with `/new`."""

    name = "new"
    description = "清空当前文字对话的上下文。"
    category = "对话"
    usage = ("new",)

    def __init__(
        self,
        service: ChatUseCase,
        tasks: ChatTaskSupervisor,
    ) -> None:
        super().__init__(service)
        self._tasks = tasks

    def definition(self) -> CommandDefinition[NoArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            NoArgumentsParser(),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=15.0),
        )

    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del arguments
        try:
            key = self._request_key(request)
            await self._tasks.cancel(key)
            await self._service.clear(key)
        except Exception as exc:
            await self._reply_request_error(request, exc)
            return
        await request.responder.reply("已开始新的对话。")


class CancelChatCommand(ChatCommandController):
    """Cancel the caller's active LLM response with `/cancel`."""

    name = "cancel"
    description = "取消当前正在生成的文字回复。"
    category = "对话"
    usage = ("cancel",)

    def __init__(
        self,
        service: ChatUseCase,
        tasks: ChatTaskSupervisor,
        *,
        active_message_reports_cancel: bool = False,
    ) -> None:
        super().__init__(service)
        self._tasks = tasks
        self._active_message_reports_cancel = active_message_reports_cancel

    def definition(self) -> CommandDefinition[NoArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            NoArgumentsParser(),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=10.0),
        )

    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del arguments
        try:
            cancelled = await self._tasks.cancel(self._request_key(request))
        except Exception as exc:
            await self._reply_request_error(request, exc)
            return
        if cancelled:
            if not self._active_message_reports_cancel:
                await request.responder.reply("已取消当前文字回复。")
        else:
            await request.responder.reply("当前没有正在生成的文字回复。")


class ModelCommand(ChatCommandController):
    """Show or change an allow-listed model with `/model [name]`."""

    name = "model"
    description = "查看或切换允许使用的模型。"
    category = "对话"
    usage = ("model [模型名称]",)

    def __init__(
        self,
        service: ChatUseCase,
        tasks: ChatTaskSupervisor,
        prefix: str = "/",
    ) -> None:
        super().__init__(service)
        self._tasks = tasks
        self._prefix = prefix

    def definition(self) -> CommandDefinition[str | None]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            ChatModelArgumentsParser(),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=30.0),
        )

    async def handle(self, request: CommandRequest, model: str | None) -> None:
        try:
            key = self._request_key(request)
            if model is None:
                status = await self._service.status(key)
                if not status.enabled:
                    await request.responder.reply("文字对话功能当前未启用。")
                    return
                await request.responder.reply(f"当前模型：{status.model}")
                return
            if self._tasks.has_active(key):
                await request.responder.reply(
                    f"当前正在生成回复；请等待完成或先使用 {self._prefix}cancel。"
                )
                return
            selected = await self._service.select_model(key, model)
        except Exception as exc:
            await self._reply_request_error(request, exc)
            return
        await request.responder.reply(f"当前模型已切换为：{selected}")


class ChatModelArgumentsParser:
    def parse(self, request: CommandRequest) -> str | None:
        assert request.text is not None
        if not request.text.tokens:
            return None
        if len(request.text.tokens) == 1:
            return request.text.tokens[0]
        raise CommandUsageError("一次只能选择一个模型。")


class ChatStatusCommand(ChatCommandController):
    """Show safe conversation metadata with `/chat-status`."""

    name = "chat-status"
    description = "查看文字对话状态，不显示聊天内容。"
    category = "对话"
    usage = ("chat-status",)

    def definition(self) -> CommandDefinition[NoArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            NoArgumentsParser(),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=10.0),
        )

    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del arguments
        try:
            status = await self._service.status(self._request_key(request))
        except Exception as exc:
            await self._reply_request_error(request, exc)
            return
        await request.responder.reply(self._status_message(status))

    @staticmethod
    def _status_message(status) -> str:
        if not status.enabled:
            return "文字对话功能当前未启用。"

        lines = [
            "文字对话状态：已启用",
            f"当前模型：{status.model}",
            f"会话：{'进行中' if status.active else '尚未开始'}",
            f"已保留消息数：{status.history_message_count}",
        ]
        if status.expires_at is not None:
            expires_at = status.expires_at.astimezone(UTC).isoformat(timespec="seconds")
            lines.append(f"会话过期时间（UTC）：{expires_at}")
        if status.cooldown_seconds > 0:
            lines.append(f"冷却剩余：{status.cooldown_seconds:.1f} 秒")
        return "\n".join(lines)
