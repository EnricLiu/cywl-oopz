from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from cywl_oopz.commands.parsing import CommandTextParser
from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.core.errors import (
    AgentInternalError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitExceeded,
)
from cywl_oopz.features.chat.commands import (
    CancelChatCommand,
    ChatCommand,
    ChatStatusCommand,
    NewConversationCommand,
)
from cywl_oopz.features.chat.history import ChatInputTooLongError
from cywl_oopz.features.chat.models import ChatResponse
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind
from cywl_oopz.features.chat.service import ChatService
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor
from cywl_oopz.integrations.oopz.chat_handlers import AmbientChatHandler, MentionChatHandler
from cywl_oopz.integrations.oopz.chat_invocation import OopzChatInvocationFactory
from cywl_oopz.integrations.oopz.command_requests import OopzCommandRequestFactory
from cywl_oopz.testing.chat import (
    InMemoryChannelSettingsRepository,
    InMemoryConversationRepository,
    RecordingChatProvider,
)
from cywl_oopz.testing.commands import dispatch_command


@dataclass
class FakeMessage:
    plain_text: str
    sender_id: str = "person-1"
    area: str = "area-1"
    channel: str = "channel-1"
    text: str = ""
    content: str = ""
    mention_list: tuple[object, ...] = ()


@dataclass
class FakeContext:
    event: object
    replies: list[str] = field(default_factory=list)

    async def reply(self, text: str):
        self.replies.append(text)
        return SimpleNamespace(message_id=f"reply-{len(self.replies)}", timestamp="123")


@dataclass
class FailingReplyContext(FakeContext):
    async def reply(self, text: str):
        del text
        raise RuntimeError("transport unavailable")


def context_for(message: FakeMessage, *, private: bool = False) -> FakeContext:
    return FakeContext(event=SimpleNamespace(message=message, is_private=private))


def test_oopz_invocation_factory_keeps_only_trusted_recipient_mentions() -> None:
    message = FakeMessage(
        "分享给朋友",
        mention_list=(
            SimpleNamespace(person="bot"),
            SimpleNamespace(person="person-1"),
            SimpleNamespace(person="friend"),
            SimpleNamespace(person="friend"),
            SimpleNamespace(person=""),
        ),
    )
    message.message_id = "source-message"
    invocation = OopzChatInvocationFactory("bot").from_context(context_for(message))

    assert invocation.mentioned_person_ids == ("friend",)
    assert invocation.source_message_id == "source-message"


def test_oopz_invocation_factory_projects_the_same_mentions_from_command_request() -> None:
    message = FakeMessage(
        "/chat 分享给朋友",
        mention_list=(
            SimpleNamespace(person="bot", is_bot=True, bot_type="CYWL"),
            SimpleNamespace(person="person-1", is_bot=False, bot_type=""),
            SimpleNamespace(person="friend", is_bot=False, bot_type=""),
        ),
    )
    message.message_id = "source-message"
    parser = CommandTextParser("/")
    request = OopzCommandRequestFactory(parser).from_message(message, context_for(message))

    assert request is not None
    invocation = OopzChatInvocationFactory("bot").from_request(request)

    assert invocation.mentioned_person_ids == ("friend",)
    assert invocation.source_message_id == "source-message"


@pytest.mark.asyncio
async def test_chat_and_new_commands_use_the_same_scoped_session(chat_settings) -> None:
    repository = InMemoryConversationRepository()
    chat = ChatService(chat_settings, RecordingChatProvider(["answer"]), repository)
    router = CommandRouter("/")
    tasks = ChatTaskSupervisor()
    router.register_definition(ChatCommand(chat, tasks).definition())
    router.register_definition(NewConversationCommand(chat, tasks).definition())
    message = FakeMessage("/chat hello")
    first_context = context_for(message)

    assert await dispatch_command(router, message, first_context) is True
    assert first_context.replies == ["answer"]
    assert repository.sessions

    reset_message = FakeMessage("/new")
    reset_context = context_for(reset_message)
    assert await dispatch_command(router, reset_message, reset_context) is True
    assert reset_context.replies == ["已开始新的对话。"]
    assert repository.sessions == {}


@pytest.mark.asyncio
async def test_chat_status_does_not_expose_message_contents(chat_settings) -> None:
    chat = ChatService(
        chat_settings, RecordingChatProvider(["sensitive answer"]), InMemoryConversationRepository()
    )
    router = CommandRouter("/")
    router.register_definition(ChatCommand(chat, ChatTaskSupervisor()).definition())
    router.register_definition(ChatStatusCommand(chat).definition())
    await dispatch_command(
        router,
        FakeMessage("/chat private question"),
        context_for(FakeMessage("/chat private question")),
    )
    status_context = context_for(FakeMessage("/chat-status"))

    await dispatch_command(router, FakeMessage("/chat-status"), status_context)

    rendered = "\n".join(status_context.replies)
    assert "private question" not in rendered
    assert "sensitive answer" not in rendered
    assert "已保留消息数：2" in rendered


@pytest.mark.asyncio
async def test_bot_mention_starts_a_scoped_conversation(chat_settings) -> None:
    repository = InMemoryConversationRepository()
    handler = MentionChatHandler(
        ChatService(chat_settings, RecordingChatProvider(["answer"]), repository), "bot"
    )
    message = FakeMessage(
        "你好",
        mention_list=(SimpleNamespace(person="bot"),),
    )
    context = context_for(message)

    consumed = await handler.handle(message, context)

    assert consumed is True
    assert context.replies == ["answer"]
    assert repository.sessions


@pytest.mark.asyncio
async def test_cancel_command_reports_when_no_response_is_running(chat_settings) -> None:
    chat = ChatService(chat_settings, RecordingChatProvider(), InMemoryConversationRepository())
    router = CommandRouter("/")
    router.register_definition(CancelChatCommand(chat, ChatTaskSupervisor()).definition())
    context = context_for(FakeMessage("/cancel"))

    await dispatch_command(router, FakeMessage("/cancel"), context)

    assert context.replies == ["当前没有正在生成的文字回复。"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["new", "cancel", "chat-status"])
async def test_no_argument_chat_commands_reject_accidental_arguments(
    chat_settings,
    name: str,
) -> None:
    chat = ChatService(
        chat_settings,
        RecordingChatProvider(),
        InMemoryConversationRepository(),
    )
    router = CommandRouter("/")
    commands = {
        "new": NewConversationCommand(chat, ChatTaskSupervisor()),
        "cancel": CancelChatCommand(chat, ChatTaskSupervisor()),
        "chat-status": ChatStatusCommand(chat),
    }
    router.register_definition(commands[name].definition())
    message = FakeMessage(f"/{name} unexpected")
    context = context_for(message)

    await dispatch_command(router, message, context)

    assert context.replies == [f"此命令不接受额外参数。\n用法：/{name}"]


@pytest.mark.asyncio
async def test_ambient_chat_accepts_private_and_explicitly_enabled_channel(chat_settings) -> None:
    repository = InMemoryConversationRepository()
    handler = AmbientChatHandler(
        ChatService(chat_settings, RecordingChatProvider(["private", "channel"]), repository),
        InMemoryChannelSettingsRepository({("area-1", "enabled-channel")}),
    )
    private_message = FakeMessage("direct message", area="", channel="private-channel")
    private_context = context_for(private_message, private=True)
    enabled_message = FakeMessage("channel message", channel="enabled-channel")
    enabled_context = context_for(enabled_message)
    disabled_message = FakeMessage("ignored", channel="disabled-channel")

    assert await handler.matches(private_message, private_context) is True
    assert await handler.handle(private_message, private_context) is True
    assert await handler.matches(enabled_message, enabled_context) is True
    assert await handler.handle(enabled_message, enabled_context) is True
    assert await handler.matches(disabled_message, context_for(disabled_message)) is False
    assert private_context.replies == ["private"]
    assert enabled_context.replies == ["channel"]


class FailingChatService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def ask(self, *_: object, **__: object):
        raise self._error


class OwnedPresentation:
    owns_message = True

    def __init__(self) -> None:
        self.events: list[ConversationProgressEvent] = []
        self.completed: ChatResponse | None = None
        self.failed = ""
        self.cancelled = False
        self.closed = False

    async def emit(self, event: ConversationProgressEvent) -> None:
        self.events.append(event)

    async def complete(self, response: ChatResponse) -> None:
        self.completed = response

    async def fail(self, message: str) -> None:
        self.failed = message

    async def cancel(self) -> None:
        self.cancelled = True

    async def aclose(self) -> None:
        self.closed = True


class OwnedPresentationFactory:
    def __init__(self, presentation: OwnedPresentation) -> None:
        self.presentation = presentation

    async def open(self, _: object) -> OwnedPresentation:
        return self.presentation


class BrokenPresentationFactory:
    async def open(self, _: object) -> OwnedPresentation:
        raise RuntimeError("presentation unavailable")


class BrokenTerminalPresentation(OwnedPresentation):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation

    async def complete(self, response: ChatResponse) -> None:
        if self.operation == "complete":
            raise RuntimeError("complete unavailable")
        await super().complete(response)

    async def fail(self, message: str) -> None:
        if self.operation == "fail":
            raise RuntimeError("fail unavailable")
        await super().fail(message)

    async def cancel(self) -> None:
        if self.operation == "cancel":
            raise RuntimeError("cancel unavailable")
        await super().cancel()

    async def aclose(self) -> None:
        if self.operation == "close":
            raise RuntimeError("close unavailable")
        await super().aclose()


class DirectPresentation(OwnedPresentation):
    owns_message = False

    def __init__(self) -> None:
        super().__init__()
        self.deliveries: list[tuple[object, ChatResponse | None, str, bool]] = []

    async def record_delivery(
        self,
        message: object,
        *,
        response: ChatResponse | None = None,
        failure_message: str = "",
        cancelled: bool = False,
    ) -> None:
        self.deliveries.append((message, response, failure_message, cancelled))


class ProgressChatService:
    enabled = True

    async def ask(self, *args: object, **kwargs: object) -> ChatResponse:
        del args
        progress = kwargs["progress"]
        await progress.emit(ConversationProgressEvent(ProgressKind.THINKING))
        return ChatResponse("只有这一条最终回答", "provider/model")


@pytest.mark.asyncio
async def test_owned_presentation_replaces_the_normal_final_reply() -> None:
    presentation = OwnedPresentation()
    router = CommandRouter("/")
    router.register_definition(
        ChatCommand(
            ProgressChatService(),
            ChatTaskSupervisor(),
            OwnedPresentationFactory(presentation),
        ).definition()
    )
    message = FakeMessage("/chat hello")
    context = context_for(message)

    await dispatch_command(router, message, context)

    assert context.replies == []
    assert [event.kind for event in presentation.events] == [ProgressKind.THINKING]
    assert presentation.completed is not None
    assert presentation.completed.content == "只有这一条最终回答"
    assert presentation.closed is True


@pytest.mark.asyncio
async def test_direct_presentation_records_the_sent_final_reply() -> None:
    presentation = DirectPresentation()
    router = CommandRouter("/")
    router.register_definition(
        ChatCommand(
            ProgressChatService(),
            ChatTaskSupervisor(),
            OwnedPresentationFactory(presentation),
        ).definition()
    )
    message = FakeMessage("/chat hello")
    context = context_for(message)

    await dispatch_command(router, message, context)

    assert context.replies == ["只有这一条最终回答"]
    assert len(presentation.deliveries) == 1
    sent, response, failure, cancelled = presentation.deliveries[0]
    assert sent.message_id == "reply-1"
    assert response is not None and response.content == "只有这一条最终回答"
    assert failure == ""
    assert cancelled is False
    assert presentation.closed is True


@pytest.mark.asyncio
async def test_owned_presentation_keeps_safe_failure_in_the_original_message() -> None:
    presentation = OwnedPresentation()
    router = CommandRouter("/")
    router.register_definition(
        ChatCommand(
            FailingChatService(ProviderTimeoutError("timeout")),
            ChatTaskSupervisor(),
            OwnedPresentationFactory(presentation),
        ).definition()
    )
    message = FakeMessage("/chat hello")
    context = context_for(message)

    await dispatch_command(router, message, context)

    assert context.replies == []
    assert presentation.failed == "模型响应超时，请稍后重试。"
    assert presentation.closed is True


@pytest.mark.asyncio
async def test_presentation_open_failure_falls_back_to_direct_reply() -> None:
    router = CommandRouter("/")
    router.register_definition(
        ChatCommand(
            ProgressChatService(),
            ChatTaskSupervisor(),
            BrokenPresentationFactory(),
        ).definition()
    )
    message = FakeMessage("/chat hello")
    context = context_for(message)

    await dispatch_command(router, message, context)

    assert context.replies == ["只有这一条最终回答"]


@pytest.mark.asyncio
async def test_direct_reply_failure_does_not_change_successful_command_outcome() -> None:
    router = CommandRouter("/")
    router.register_definition(
        ChatCommand(ProgressChatService(), ChatTaskSupervisor()).definition()
    )
    message = FakeMessage("/chat hello")
    context = FailingReplyContext(event=SimpleNamespace(message=message, is_private=False))

    await dispatch_command(router, message, context)

    assert context.replies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["complete", "fail", "close"])
async def test_terminal_presentation_failure_does_not_replace_primary_outcome(
    operation: str,
) -> None:
    presentation = BrokenTerminalPresentation(operation)
    service = (
        FailingChatService(ProviderTimeoutError("timeout"))
        if operation == "fail"
        else ProgressChatService()
    )
    router = CommandRouter("/")
    router.register_definition(
        ChatCommand(
            service,
            ChatTaskSupervisor(),
            OwnedPresentationFactory(presentation),
        ).definition()
    )
    message = FakeMessage("/chat hello")
    context = context_for(message)

    await dispatch_command(router, message, context)

    if operation == "complete":
        assert context.replies == ["只有这一条最终回答"]
    elif operation == "fail":
        assert context.replies == ["模型响应超时，请稍后重试。"]
    else:
        assert context.replies == []
        assert presentation.completed is not None


@pytest.mark.asyncio
async def test_mention_presentation_open_failure_falls_back_to_direct_reply() -> None:
    handler = MentionChatHandler(
        ProgressChatService(),
        "bot",
        BrokenPresentationFactory(),
    )
    message = FakeMessage(
        "hello",
        mention_list=(SimpleNamespace(person="bot"),),
    )
    context = context_for(message)

    assert await handler.handle(message, context) is True

    assert context.replies == ["只有这一条最终回答"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["complete", "fail", "close"])
async def test_mention_terminal_presentation_failure_preserves_primary_outcome(
    operation: str,
) -> None:
    presentation = BrokenTerminalPresentation(operation)
    service = (
        FailingChatService(ProviderTimeoutError("timeout"))
        if operation == "fail"
        else ProgressChatService()
    )
    handler = MentionChatHandler(
        service,
        "bot",
        OwnedPresentationFactory(presentation),
    )
    message = FakeMessage(
        "hello",
        mention_list=(SimpleNamespace(person="bot"),),
    )
    context = context_for(message)

    assert await handler.handle(message, context) is True

    if operation == "complete":
        assert context.replies == ["只有这一条最终回答"]
    elif operation == "fail":
        assert context.replies == ["模型响应超时，请稍后重试。"]
    else:
        assert context.replies == []
        assert presentation.completed is not None


@pytest.mark.asyncio
async def test_mention_live_and_fallback_reply_failure_are_contained() -> None:
    presentation = BrokenTerminalPresentation("complete")
    handler = MentionChatHandler(
        ProgressChatService(),
        "bot",
        OwnedPresentationFactory(presentation),
    )
    message = FakeMessage(
        "hello",
        mention_list=(SimpleNamespace(person="bot"),),
    )
    context = FailingReplyContext(event=SimpleNamespace(message=message, is_private=False))

    assert await handler.handle(message, context) is True

    assert context.replies == []
    assert presentation.closed is True


@pytest.mark.asyncio
async def test_cancel_updates_owned_message_without_a_second_command_reply() -> None:
    presentation = OwnedPresentation()
    supervisor = ChatTaskSupervisor()
    started = asyncio.Event()

    class WaitingService:
        enabled = True

        async def ask(self, *args: object, **kwargs: object) -> ChatResponse:
            del args, kwargs
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    service = WaitingService()
    chat = ChatCommand(service, supervisor, OwnedPresentationFactory(presentation))
    source_message = FakeMessage("/chat wait")
    source_context = context_for(source_message)
    router = CommandRouter("/")
    router.register_definition(chat.definition())
    router.register_definition(
        CancelChatCommand(
            service,
            supervisor,
            active_message_reports_cancel=True,
        ).definition()
    )
    operation = asyncio.create_task(dispatch_command(router, source_message, source_context))
    await asyncio.wait_for(started.wait(), timeout=1)

    cancel_context = context_for(FakeMessage("/cancel"))
    await dispatch_command(router, cancel_context.event.message, cancel_context)
    await asyncio.gather(operation, return_exceptions=True)

    assert presentation.cancelled is True
    assert presentation.closed is True
    assert source_context.replies == []
    assert cancel_context.replies == []


@pytest.mark.asyncio
async def test_cancel_presentation_failure_preserves_cancellation() -> None:
    presentation = BrokenTerminalPresentation("cancel")
    supervisor = ChatTaskSupervisor()
    started = asyncio.Event()

    class WaitingService:
        enabled = True

        async def ask(self, *args: object, **kwargs: object) -> ChatResponse:
            del args, kwargs
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    service = WaitingService()
    source_message = FakeMessage("/chat wait")
    source_context = context_for(source_message)
    router = CommandRouter("/")
    router.register_definition(
        ChatCommand(
            service,
            supervisor,
            OwnedPresentationFactory(presentation),
        ).definition()
    )
    router.register_definition(
        CancelChatCommand(
            service,
            supervisor,
            active_message_reports_cancel=True,
        ).definition()
    )
    operation = asyncio.create_task(dispatch_command(router, source_message, source_context))
    await asyncio.wait_for(started.wait(), timeout=1)

    cancel_context = context_for(FakeMessage("/cancel"))
    await dispatch_command(router, cancel_context.event.message, cancel_context)
    outcomes = await asyncio.gather(operation, return_exceptions=True)

    assert isinstance(outcomes[0], asyncio.CancelledError)
    assert source_context.replies == ["已取消当前文字回复。"]
    assert cancel_context.replies == []


@pytest.mark.asyncio
async def test_mention_cancel_presentation_failure_preserves_cancellation() -> None:
    presentation = BrokenTerminalPresentation("cancel")
    started = asyncio.Event()

    class WaitingService:
        enabled = True

        async def ask(self, *args: object, **kwargs: object) -> ChatResponse:
            del args, kwargs
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    handler = MentionChatHandler(
        WaitingService(),
        "bot",
        OwnedPresentationFactory(presentation),
    )
    message = FakeMessage(
        "wait",
        mention_list=(SimpleNamespace(person="bot"),),
    )
    context = context_for(message)
    operation = asyncio.create_task(handler.handle(message, context))
    await asyncio.wait_for(started.wait(), timeout=1)

    operation.cancel()
    outcomes = await asyncio.gather(operation, return_exceptions=True)

    assert isinstance(outcomes[0], asyncio.CancelledError)
    assert context.replies == ["已取消当前文字回复。"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderTimeoutError("timeout"), "模型响应超时，请稍后重试。"),
        (ProviderResponseError("invalid"), "模型服务暂时不可用，请稍后重试。"),
        (RateLimitExceeded("global concurrency"), "当前对话请求较多，请稍后重试。"),
        (ChatInputTooLongError(), "这条消息太长，请缩短后再试。"),
    ],
)
async def test_chat_command_maps_expected_failures_to_one_safe_reply(error, expected) -> None:
    router = CommandRouter("/")
    router.register_definition(
        ChatCommand(FailingChatService(error), ChatTaskSupervisor()).definition()
    )
    message = FakeMessage("/chat hello")
    context = context_for(message)

    await dispatch_command(router, message, context)

    assert context.replies == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ValueError("internal invariant"), AgentInternalError("adapter invariant")],
)
async def test_chat_command_internal_failure_has_safe_reference(error: Exception) -> None:
    router = CommandRouter("/")
    router.register_definition(
        ChatCommand(FailingChatService(error), ChatTaskSupervisor()).definition()
    )
    message = FakeMessage("/chat hello")
    context = context_for(message)

    await dispatch_command(router, message, context)

    assert len(context.replies) == 1
    assert context.replies[0].startswith("这次处理在内部出错，请稍后重试（参考号：")
    assert context.replies[0].endswith("）。")
    assert "invariant" not in context.replies[0]


@pytest.mark.asyncio
async def test_mention_handler_uses_the_same_internal_error_protocol() -> None:
    handler = MentionChatHandler(FailingChatService(ValueError("private detail")), "bot")
    message = FakeMessage(
        "hello",
        mention_list=(SimpleNamespace(person="bot"),),
    )
    context = context_for(message)

    assert await handler.handle(message, context) is True

    assert len(context.replies) == 1
    assert context.replies[0].startswith("这次处理在内部出错，请稍后重试（参考号：")
    assert "private detail" not in context.replies[0]
