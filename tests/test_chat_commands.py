from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.core.errors import ProviderResponseError, ProviderTimeoutError, RateLimitExceeded
from cywl_oopz.features.chat.commands import (
    AmbientChatHandler,
    CancelChatCommand,
    ChatCommand,
    ChatStatusCommand,
    MentionChatHandler,
    NewConversationCommand,
)
from cywl_oopz.features.chat.models import ChatResponse, ConversationKey
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind
from cywl_oopz.features.chat.service import ChatService
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor
from cywl_oopz.testing.chat import (
    InMemoryChannelSettingsRepository,
    InMemoryConversationRepository,
    RecordingChatProvider,
)


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

    async def reply(self, text: str) -> None:
        self.replies.append(text)


def context_for(message: FakeMessage, *, private: bool = False) -> FakeContext:
    return FakeContext(event=SimpleNamespace(message=message, is_private=private))


@pytest.mark.asyncio
async def test_chat_and_new_commands_use_the_same_scoped_session(chat_settings) -> None:
    repository = InMemoryConversationRepository()
    chat = ChatService(chat_settings, RecordingChatProvider(["answer"]), repository)
    router = CommandRouter("!")
    router.register(ChatCommand(chat))
    router.register(NewConversationCommand(chat, ChatTaskSupervisor()))
    message = FakeMessage("!chat hello")
    first_context = context_for(message)

    assert await router.dispatch(message, first_context) is True
    assert first_context.replies == ["answer"]
    assert repository.sessions

    reset_message = FakeMessage("!new")
    reset_context = context_for(reset_message)
    assert await router.dispatch(reset_message, reset_context) is True
    assert reset_context.replies == ["已开始新的对话。"]
    assert repository.sessions == {}


@pytest.mark.asyncio
async def test_chat_status_does_not_expose_message_contents(chat_settings) -> None:
    chat = ChatService(
        chat_settings, RecordingChatProvider(["sensitive answer"]), InMemoryConversationRepository()
    )
    router = CommandRouter("!")
    router.register(ChatCommand(chat))
    router.register(ChatStatusCommand(chat))
    await router.dispatch(
        FakeMessage("!chat private question"), context_for(FakeMessage("!chat private question"))
    )
    status_context = context_for(FakeMessage("!chat-status"))

    await router.dispatch(FakeMessage("!chat-status"), status_context)

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
    router = CommandRouter("!")
    router.register(CancelChatCommand(chat, ChatTaskSupervisor()))
    context = context_for(FakeMessage("!cancel"))

    await router.dispatch(FakeMessage("!cancel"), context)

    assert context.replies == ["当前没有正在生成的文字回复。"]


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
    router = CommandRouter("!")
    router.register(
        ChatCommand(
            ProgressChatService(),
            OwnedPresentationFactory(presentation),
        )
    )
    message = FakeMessage("!chat hello")
    context = context_for(message)

    await router.dispatch(message, context)

    assert context.replies == []
    assert [event.kind for event in presentation.events] == [ProgressKind.THINKING]
    assert presentation.completed is not None
    assert presentation.completed.content == "只有这一条最终回答"
    assert presentation.closed is True


@pytest.mark.asyncio
async def test_owned_presentation_keeps_safe_failure_in_the_original_message() -> None:
    presentation = OwnedPresentation()
    router = CommandRouter("!")
    router.register(
        ChatCommand(
            FailingChatService(ProviderTimeoutError("timeout")),
            OwnedPresentationFactory(presentation),
        )
    )
    message = FakeMessage("!chat hello")
    context = context_for(message)

    await router.dispatch(message, context)

    assert context.replies == []
    assert presentation.failed == "模型响应超时，请稍后重试。"
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
    chat = ChatCommand(service, OwnedPresentationFactory(presentation))
    source_message = FakeMessage("!chat wait")
    source_context = context_for(source_message)
    key = ConversationKey.from_oopz_context(source_context)
    operation = chat.execute(
        CommandRouter("!").parse(source_message.plain_text),
        source_context,
    )
    assert supervisor.start(key, operation) is True
    await asyncio.wait_for(started.wait(), timeout=1)

    cancel_context = context_for(FakeMessage("!cancel"))
    cancel = CancelChatCommand(
        service,
        supervisor,
        active_message_reports_cancel=True,
    )
    await cancel.execute(SimpleNamespace(), cancel_context)

    assert presentation.cancelled is True
    assert presentation.closed is True
    assert source_context.replies == []
    assert cancel_context.replies == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderTimeoutError("timeout"), "模型响应超时，请稍后重试。"),
        (ProviderResponseError("invalid"), "模型服务暂时不可用，请稍后重试。"),
        (RateLimitExceeded("global concurrency"), "当前对话请求较多，请稍后重试。"),
    ],
)
async def test_chat_command_maps_expected_failures_to_one_safe_reply(error, expected) -> None:
    router = CommandRouter("!")
    router.register(ChatCommand(FailingChatService(error)))
    message = FakeMessage("!chat hello")
    context = context_for(message)

    await router.dispatch(message, context)

    assert context.replies == [expected]
