from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.features.access.models import AccessRole, RoleBinding, RoleBindingScope
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.features.admin.actions import RecallMessageAction
from cywl_oopz.features.admin.commands import RecallCommand
from cywl_oopz.features.admin.models import (
    MessageRecallOutcome,
    OopzMessageAddress,
    OopzMessageScope,
    OutboundMessageKind,
    OutboundMessageReceipt,
    OutboundMessageState,
    ReferencedMessageCandidate,
)
from cywl_oopz.features.admin.recall import (
    BotMessageRecallTransportError,
    MessageRecallService,
)
from cywl_oopz.features.admin.references import ReferencedMessageResolver
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor, OutboundChatTaskCanceller
from cywl_oopz.integrations.oopz.message_recall import (
    OopzBotMessageRecallGateway,
    OopzRecentBotMessageLookup,
    OopzReferencedMessageParser,
)
from cywl_oopz.testing.commands import dispatch_command

ADDRESS = OopzMessageAddress(OopzMessageScope.CHANNEL, "area", "channel")


def receipt(state: OutboundMessageState = OutboundMessageState.FINAL) -> OutboundMessageReceipt:
    return OutboundMessageReceipt(
        "bot-message",
        "123",
        OutboundMessageKind.AGENT_RESPONSE,
        state,
        ADDRESS,
        owner_person_id="person",
    )


@dataclass
class ReceiptRepository:
    current: OutboundMessageReceipt | None
    events: list[str] = field(default_factory=list)
    created: list[OutboundMessageReceipt] = field(default_factory=list)

    async def get_by_message(self, message_id, address):
        self.events.append("get")
        value = self.current
        if value is None or value.message_id != message_id or value.address != address:
            return None
        return value

    async def create(self, value: OutboundMessageReceipt) -> bool:
        self.events.append("create")
        self.created.append(value)
        if self.current is not None:
            return False
        self.current = value
        return True

    async def mark_recalled(self, message_id: str) -> bool:
        self.events.append("mark")
        if (
            self.current is None
            or self.current.message_id != message_id
            or self.current.state is OutboundMessageState.RECALLED
        ):
            return False
        self.current = replace(self.current, state=OutboundMessageState.RECALLED)
        return True


@dataclass
class RecentLookup:
    value: ReferencedMessageCandidate | None = None
    calls: int = 0

    async def find(self, message_id, address):
        del message_id, address
        self.calls += 1
        return self.value


def candidate(*, sender: str = "bot") -> ReferencedMessageCandidate:
    return ReferencedMessageCandidate("bot-message", "123", sender, ADDRESS)


@pytest.mark.asyncio
async def test_reference_resolver_prefers_exact_receipt_without_history() -> None:
    repository = ReceiptRepository(receipt())
    recent = RecentLookup(candidate())
    resolver = ReferencedMessageResolver(repository, recent, "bot")

    result = await resolver.resolve("bot-message", ADDRESS, candidate(sender="someone-else"))

    assert result == receipt()
    assert recent.calls == 0
    assert repository.created == []


@pytest.mark.asyncio
async def test_reference_resolver_materializes_strict_legacy_candidate() -> None:
    repository = ReceiptRepository(None)
    resolver = ReferencedMessageResolver(repository, RecentLookup(), "bot")

    result = await resolver.resolve("bot-message", ADDRESS, candidate())

    assert result is not None
    assert result.kind is OutboundMessageKind.COMMAND_REPLY
    assert result.address == ADDRESS
    assert repository.current == result


@pytest.mark.asyncio
async def test_reference_resolver_rejects_wrong_sender_and_address() -> None:
    repository = ReceiptRepository(None)
    wrong_address = ReferencedMessageCandidate(
        "bot-message",
        "123",
        "bot",
        OopzMessageAddress(OopzMessageScope.CHANNEL, "other-area", "channel"),
    )
    resolver = ReferencedMessageResolver(repository, RecentLookup(wrong_address), "bot")

    result = await resolver.resolve("bot-message", ADDRESS, candidate(sender="user"))

    assert result is None
    assert repository.created == []


def test_oopz_reference_parser_accepts_sdk_shape_and_mapping_aliases() -> None:
    sdk_value = SimpleNamespace(
        message_id="bot-message",
        timestamp="123",
        sender_id="bot",
        area="area",
        channel="channel",
        target="",
    )
    mapping = {
        "messageId": "private-message",
        "timestamp": "456",
        "person": "bot",
        "area": "",
        "channel": "private-channel",
        "target": "person",
    }

    channel = OopzReferencedMessageParser.parse(sdk_value)
    private = OopzReferencedMessageParser.parse(mapping)

    assert channel == candidate()
    assert private is not None
    assert private.address == OopzMessageAddress(
        OopzMessageScope.PRIVATE,
        "",
        "private-channel",
        "person",
    )
    assert OopzReferencedMessageParser.parse({"messageId": "incomplete"}) is None


@pytest.mark.asyncio
async def test_recent_lookup_is_bounded_and_channel_only() -> None:
    class Messages:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def get_channel_messages(self, **kwargs):
            self.calls.append(kwargs)
            return [
                SimpleNamespace(
                    message_id="bot-message",
                    timestamp="123",
                    sender_id="bot",
                    area="area",
                    channel="channel",
                    target="",
                )
            ]

    messages = Messages()
    lookup = OopzRecentBotMessageLookup(SimpleNamespace(messages=messages))

    found = await lookup.find("bot-message", ADDRESS)
    private = await lookup.find(
        "bot-message",
        OopzMessageAddress(OopzMessageScope.PRIVATE, "", "private", "person"),
    )

    assert found == candidate()
    assert private is None
    assert messages.calls == [{"area": "area", "channel": "channel", "size": 50}]


@dataclass
class SequenceAction:
    name: str
    events: list[str]

    async def dismiss(self, message_id: str) -> bool:
        del message_id
        self.events.append(self.name)
        return True

    async def cancel_for_message(self, value: OutboundMessageReceipt) -> bool:
        del value
        self.events.append(self.name)
        return True

    async def recall(self, value: OutboundMessageReceipt) -> None:
        del value
        self.events.append(self.name)


@pytest.mark.asyncio
async def test_recall_service_orders_dismiss_cancel_transport_and_state() -> None:
    events: list[str] = []
    repository = ReceiptRepository(receipt(), events)
    service = MessageRecallService(
        ReferencedMessageResolver(repository, RecentLookup(), "bot"),
        repository,
        SequenceAction("dismiss", events),
        SequenceAction("cancel", events),
        SequenceAction("recall", events),
    )

    outcome = await service.recall("bot-message", ADDRESS)

    assert outcome is MessageRecallOutcome.RECALLED
    assert events == ["get", "dismiss", "cancel", "recall", "mark"]
    assert repository.current is not None
    assert repository.current.state is OutboundMessageState.RECALLED


@pytest.mark.asyncio
async def test_recalling_historical_message_preserves_current_conversation_task() -> None:
    events: list[str] = []
    repository = ReceiptRepository(receipt(), events)
    tasks = ChatTaskSupervisor()
    key = ConversationKey("channel", "area", "channel", "person")
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def current_agent_loop() -> None:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    assert tasks.start(key, current_agent_loop())
    await asyncio.wait_for(started.wait(), timeout=1)

    class HistoricalPresentation:
        async def dismiss(self, message_id: str) -> bool:
            del message_id
            events.append("dismiss")
            return False

    service = MessageRecallService(
        ReferencedMessageResolver(repository, RecentLookup(), "bot"),
        repository,
        HistoricalPresentation(),
        OutboundChatTaskCanceller(tasks),
        SequenceAction("recall", events),
    )

    outcome = await service.recall("bot-message", ADDRESS)

    assert outcome is MessageRecallOutcome.RECALLED
    assert events == ["get", "dismiss", "recall", "mark"]
    assert repository.current is not None
    assert repository.current.state is OutboundMessageState.RECALLED
    assert tasks.has_active(key)
    assert not cancelled.is_set()

    release.set()
    await tasks.wait(key)


@pytest.mark.asyncio
async def test_recall_service_is_idempotent_under_concurrency() -> None:
    repository = ReceiptRepository(receipt())
    transport = SequenceAction("recall", repository.events)
    service = MessageRecallService(
        ReferencedMessageResolver(repository, RecentLookup(), "bot"),
        repository,
        SequenceAction("dismiss", repository.events),
        SequenceAction("cancel", repository.events),
        transport,
    )

    outcomes = await asyncio.gather(
        service.recall("bot-message", ADDRESS),
        service.recall("bot-message", ADDRESS),
    )

    assert outcomes == [MessageRecallOutcome.RECALLED, MessageRecallOutcome.ALREADY_RECALLED]
    assert repository.events.count("recall") == 1
    assert repository.events.count("mark") == 1


@pytest.mark.asyncio
async def test_recall_transport_failure_does_not_mark_receipt() -> None:
    class FailureGateway:
        async def recall(self, value) -> None:
            del value
            raise BotMessageRecallTransportError("rejected")

    repository = ReceiptRepository(receipt())
    service = MessageRecallService(
        ReferencedMessageResolver(repository, RecentLookup(), "bot"),
        repository,
        SequenceAction("dismiss", repository.events),
        SequenceAction("cancel", repository.events),
        FailureGateway(),
    )

    with pytest.raises(BotMessageRecallTransportError):
        await service.recall("bot-message", ADDRESS)

    assert repository.current is not None
    assert repository.current.state is OutboundMessageState.FINAL
    assert "mark" not in repository.events


@pytest.mark.asyncio
async def test_oopz_recall_gateway_uses_channel_and_private_endpoints() -> None:
    class Messages:
        def __init__(self) -> None:
            self.channel: list[dict[str, object]] = []
            self.private: list[dict[str, object]] = []

        async def recall_message(self, **kwargs):
            self.channel.append(kwargs)
            return SimpleNamespace(ok=True)

        async def recall_private_message(self, **kwargs):
            self.private.append(kwargs)
            return SimpleNamespace(ok=True)

    messages = Messages()
    gateway = OopzBotMessageRecallGateway(SimpleNamespace(messages=messages))
    private_receipt = replace(
        receipt(),
        message_id="private-message",
        address=OopzMessageAddress(
            OopzMessageScope.PRIVATE,
            "",
            "private-channel",
            "person",
        ),
    )

    await gateway.recall(receipt())
    await gateway.recall(private_receipt)

    assert messages.channel == [
        {
            "message_id": "bot-message",
            "area": "area",
            "channel": "channel",
            "timestamp": "123",
        }
    ]
    assert messages.private == [
        {
            "message_id": "private-message",
            "channel": "private-channel",
            "target": "person",
            "timestamp": "123",
        }
    ]


@pytest.mark.asyncio
async def test_outbound_task_canceller_reconstructs_channel_and_private_keys() -> None:
    class Tasks:
        def __init__(self) -> None:
            self.keys: list[ConversationKey] = []

        async def cancel(self, key: ConversationKey) -> bool:
            self.keys.append(key)
            return True

    tasks = Tasks()
    canceller = OutboundChatTaskCanceller(tasks)  # type: ignore[arg-type]
    private_receipt = replace(
        receipt(),
        address=OopzMessageAddress(
            OopzMessageScope.PRIVATE,
            "",
            "private-channel",
            "person",
        ),
    )

    assert await canceller.cancel_for_message(receipt())
    assert await canceller.cancel_for_message(private_receipt)
    assert not await canceller.cancel_for_message(replace(receipt(), owner_person_id=""))

    assert tasks.keys == [
        ConversationKey("channel", "area", "channel", "person"),
        ConversationKey("private", "", "", "person"),
    ]


@dataclass
class RoleRepository:
    records: tuple[RoleBinding, ...]

    async def list_for_subject(self, person_id: str) -> tuple[RoleBinding, ...]:
        return tuple(item for item in self.records if item.subject_person_id == person_id)


class RecallUseCase:
    def __init__(self, outcome: MessageRecallOutcome = MessageRecallOutcome.RECALLED) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, OopzMessageAddress, object]] = []

    async def recall(self, message_id, address, embedded=None):
        self.calls.append((message_id, address, embedded))
        return self.outcome


class CommandMessage:
    def __init__(self, text: str = "/recall") -> None:
        self.plain_text = text
        self.text = text
        self.content = text
        self.sender_id = "moderator"
        self.area = "area"
        self.channel = "channel"
        self.reference_message_id = "bot-message"
        self.reference_message = None


class CommandContext:
    def __init__(self, message: CommandMessage, *, reaction_fails: bool = False) -> None:
        self.event = SimpleNamespace(message=message, is_private=False)
        self.reaction_fails = reaction_fails
        self.reactions: list[str] = []
        self.replies: list[str] = []

    async def react(self, emoji: str):
        if self.reaction_fails:
            raise RuntimeError("reaction failed")
        self.reactions.append(emoji)
        return SimpleNamespace(ok=True)

    async def reply(self, text: str) -> None:
        self.replies.append(text)


def recall_router(use_case: RecallUseCase, *, allowed: bool = True) -> CommandRouter:
    records = (
        (
            RoleBinding(
                "moderator",
                AccessRole.MODERATOR,
                RoleBindingScope.CHANNEL,
                area_id="area",
                channel_id="channel",
            ),
        )
        if allowed
        else ()
    )
    router = CommandRouter("/", AuthorizationService(RoleRepository(records)))
    router.register_definition(
        RecallCommand(RecallMessageAction(use_case)).definition()  # type: ignore[arg-type]
    )
    return router


@pytest.mark.asyncio
async def test_recall_command_requires_permission_and_confirms_with_reaction() -> None:
    use_case = RecallUseCase()
    message = CommandMessage()
    context = CommandContext(message)

    assert await dispatch_command(recall_router(use_case), message, context)

    assert context.reactions == ["✅"]
    assert context.replies == []
    assert use_case.calls[0][:2] == ("bot-message", ADDRESS)

    denied = RecallUseCase()
    denied_context = CommandContext(CommandMessage())
    await dispatch_command(
        recall_router(denied, allowed=False), denied_context.event.message, denied_context
    )
    assert denied.calls == []
    assert denied_context.replies == ["你没有执行此操作的权限。"]


@pytest.mark.asyncio
async def test_recall_command_stays_silent_when_confirmation_reaction_fails() -> None:
    use_case = RecallUseCase()
    message = CommandMessage()
    context = CommandContext(message, reaction_fails=True)

    await dispatch_command(recall_router(use_case), message, context)

    assert context.reactions == []
    assert context.replies == []


@pytest.mark.asyncio
async def test_recall_command_reports_idempotency_and_rejects_raw_message_id() -> None:
    already = RecallUseCase(MessageRecallOutcome.ALREADY_RECALLED)
    already_message = CommandMessage()
    already_context = CommandContext(already_message)

    await dispatch_command(recall_router(already), already_message, already_context)

    assert already_context.reactions == []
    assert already_context.replies == ["这条回复已经撤回。"]

    invalid = RecallUseCase()
    invalid_message = CommandMessage("/recall bot-message")
    invalid_context = CommandContext(invalid_message)
    await dispatch_command(recall_router(invalid), invalid_message, invalid_context)

    assert invalid.calls == []
    assert invalid_context.replies == ["用法：/recall（请引用一条 CYWL 回复）"]
