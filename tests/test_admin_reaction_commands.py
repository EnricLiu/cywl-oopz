from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from cywl_oopz.features.access.models import (
    AccessPrincipal,
    AccessResource,
    AccessRole,
    RoleBinding,
    RoleBindingScope,
)
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.features.admin.actions import DebugMessageAction, RecallMessageAction
from cywl_oopz.features.admin.models import (
    MessageRecallOutcome,
    OopzMessageAddress,
    OopzMessageScope,
    OutboundMessageKind,
    OutboundMessageReceipt,
    OutboundMessageState,
)
from cywl_oopz.features.admin.reaction_commands import (
    DebugReactionCommand,
    ReactionCommandInvocation,
    ReactionCommandRouter,
    RecallReactionCommand,
)
from cywl_oopz.integrations.oopz.editable_messages import EditableMessageRef
from cywl_oopz.integrations.oopz.reaction_commands import (
    OopzReactionCommandInvocationParser,
    OopzReactionCommandResponder,
)

ADDRESS = OopzMessageAddress(OopzMessageScope.CHANNEL, "area", "channel")


@dataclass
class RoleRepository:
    records: tuple[RoleBinding, ...]

    async def list_for_subject(self, person_id: str) -> tuple[RoleBinding, ...]:
        return tuple(record for record in self.records if record.subject_person_id == person_id)


class RecordingResponder:
    def __init__(self) -> None:
        self.messages: list[tuple[ReactionCommandInvocation, str]] = []

    async def send(self, invocation: ReactionCommandInvocation, text: str) -> None:
        self.messages.append((invocation, text))


class TargetResolver:
    def __init__(self, tracked: bool = True) -> None:
        self.value = (
            OutboundMessageReceipt(
                "bot-message",
                "123",
                OutboundMessageKind.AGENT_RESPONSE,
                OutboundMessageState.FINAL,
                ADDRESS,
            )
            if tracked
            else None
        )
        self.calls: list[tuple[str, OopzMessageAddress]] = []

    async def resolve(self, message_id, address, embedded=None):
        del embedded
        self.calls.append((message_id, address))
        return self.value


class RecallService:
    def __init__(self, outcome: MessageRecallOutcome = MessageRecallOutcome.RECALLED) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, OopzMessageAddress]] = []

    async def recall(self, message_id, address, embedded=None):
        del embedded
        self.calls.append((message_id, address))
        return self.outcome


class DiagnosticRepository:
    def __init__(self, diagnostic: object | None) -> None:
        self.diagnostic = diagnostic
        self.calls: list[tuple[str, OopzMessageAddress]] = []

    async def get_by_outbound_message(self, message_id, address):
        self.calls.append((message_id, address))
        return self.diagnostic


class DiagnosticRenderer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, bool]] = []

    def render(self, diagnostic, *, verbose):
        self.calls.append((diagnostic, verbose))
        return ("debug-page-1", "debug-page-2")


def invocation(emoji: str, person_id: str = "admin") -> ReactionCommandInvocation:
    return ReactionCommandInvocation(
        emoji,
        "bot-message",
        AccessPrincipal(person_id),
        AccessResource.channel("area", "channel"),
        ADDRESS,
    )


def authorizer(role: AccessRole | None) -> AuthorizationService:
    records = (
        (
            RoleBinding(
                "admin",
                role,
                RoleBindingScope.CHANNEL,
                area_id="area",
                channel_id="channel",
            ),
        )
        if role is not None
        else ()
    )
    return AuthorizationService(RoleRepository(records))


def test_oopz_reaction_parser_accepts_additions_and_ignores_withdrawals_and_self() -> None:
    context = SimpleNamespace(config=SimpleNamespace(person_uid="bot"))
    event = SimpleNamespace(
        type="REPLY",
        person="admin",
        message_id="bot-message",
        area="area",
        channel="channel",
        emoji="🤯",
    )

    parsed = OopzReactionCommandInvocationParser.parse(context, event)

    assert parsed == invocation("🤯")
    event.emoji = "129327"
    assert OopzReactionCommandInvocationParser.parse(context, event) == invocation("🤯")
    event.type = "WITHDRAW"
    assert OopzReactionCommandInvocationParser.parse(context, event) is None
    event.type = "REPLY"
    event.person = "bot"
    assert OopzReactionCommandInvocationParser.parse(context, event) is None


@pytest.mark.asyncio
async def test_recall_reaction_checks_permission_and_recalls_exact_message() -> None:
    recall = RecallService()
    responder = RecordingResponder()
    router = ReactionCommandRouter(
        authorizer(AccessRole.MODERATOR),
        TargetResolver(),  # type: ignore[arg-type]
        responder,
    )
    router.register(
        RecallReactionCommand(RecallMessageAction(recall))  # type: ignore[arg-type]
    )

    consumed = await router.dispatch(invocation("🫥"))

    assert consumed is True
    assert recall.calls == [("bot-message", ADDRESS)]
    assert responder.messages == []


@pytest.mark.asyncio
async def test_recall_reaction_treats_already_recalled_as_silent_idempotent_success() -> None:
    recall = RecallService(MessageRecallOutcome.ALREADY_RECALLED)
    responder = RecordingResponder()
    router = ReactionCommandRouter(
        authorizer(AccessRole.MODERATOR),
        TargetResolver(),  # type: ignore[arg-type]
        responder,
    )
    router.register(
        RecallReactionCommand(RecallMessageAction(recall))  # type: ignore[arg-type]
    )

    assert await router.dispatch(invocation("🫥")) is True

    assert recall.calls == [("bot-message", ADDRESS)]
    assert responder.messages == []


@pytest.mark.asyncio
async def test_debug_reaction_renders_bounded_normal_diagnostics() -> None:
    diagnostic = object()
    repository = DiagnosticRepository(diagnostic)
    renderer = DiagnosticRenderer()
    responder = RecordingResponder()
    router = ReactionCommandRouter(
        authorizer(AccessRole.ADMIN),
        TargetResolver(),  # type: ignore[arg-type]
        responder,
    )
    router.register(
        DebugReactionCommand(
            DebugMessageAction(repository, renderer)  # type: ignore[arg-type]
        )
    )

    consumed = await router.dispatch(invocation("🤯"))

    assert consumed is True
    assert repository.calls == [("bot-message", ADDRESS)]
    assert renderer.calls == [(diagnostic, False)]
    assert [message for _, message in responder.messages] == [
        "debug-page-1",
        "debug-page-2",
    ]


@pytest.mark.asyncio
async def test_reaction_command_denial_happens_before_action() -> None:
    recall = RecallService()
    responder = RecordingResponder()
    router = ReactionCommandRouter(
        authorizer(None),
        TargetResolver(),  # type: ignore[arg-type]
        responder,
    )
    router.register(
        RecallReactionCommand(RecallMessageAction(recall))  # type: ignore[arg-type]
    )

    consumed = await router.dispatch(invocation("🫥"))

    assert consumed is True
    assert recall.calls == []
    assert [message for _, message in responder.messages] == ["你没有执行此操作的权限。"]


@pytest.mark.asyncio
async def test_unmapped_reaction_is_ignored() -> None:
    responder = RecordingResponder()
    router = ReactionCommandRouter(
        authorizer(AccessRole.ADMIN),
        TargetResolver(),  # type: ignore[arg-type]
        responder,
    )

    assert await router.dispatch(invocation("👍")) is False
    assert responder.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("emoji", ["🫥", "🤯"])
async def test_reaction_on_non_bot_message_is_silent(emoji: str) -> None:
    recall = RecallService()
    repository = DiagnosticRepository(object())
    responder = RecordingResponder()
    router = ReactionCommandRouter(
        authorizer(None),
        TargetResolver(tracked=False),  # type: ignore[arg-type]
        responder,
    )
    router.register(
        RecallReactionCommand(RecallMessageAction(recall))  # type: ignore[arg-type]
    )
    router.register(
        DebugReactionCommand(
            DebugMessageAction(repository, DiagnosticRenderer())  # type: ignore[arg-type]
        )
    )

    assert await router.dispatch(invocation(emoji)) is True

    assert recall.calls == []
    assert repository.calls == []
    assert responder.messages == []


@pytest.mark.asyncio
async def test_tracked_non_agent_message_debug_reaction_is_silent() -> None:
    repository = DiagnosticRepository(None)
    responder = RecordingResponder()
    router = ReactionCommandRouter(
        authorizer(AccessRole.ADMIN),
        TargetResolver(),  # type: ignore[arg-type]
        responder,
    )
    router.register(
        DebugReactionCommand(
            DebugMessageAction(repository, DiagnosticRenderer())  # type: ignore[arg-type]
        )
    )

    assert await router.dispatch(invocation("🤯")) is True

    assert repository.calls == [("bot-message", ADDRESS)]
    assert responder.messages == []


@pytest.mark.asyncio
async def test_oopz_reaction_responder_tracks_debug_pages_as_command_replies() -> None:
    class EditableMessages:
        def __init__(self) -> None:
            self.created: list[tuple[object, str]] = []
            self.tracked: list[tuple[EditableMessageRef, dict[str, object]]] = []

        async def create_reply(self, address, text):
            self.created.append((address, text))
            return EditableMessageRef(
                "debug-message",
                "123",
                address.scope,
                address.area_id,
                address.channel_id,
                address.target_person_id,
                address.reference_message_id,
            )

        async def track_created(self, message, **kwargs):
            self.tracked.append((message, kwargs))

    messages = EditableMessages()
    responder = OopzReactionCommandResponder(messages)  # type: ignore[arg-type]

    await responder.send(invocation("🤯"), "debug-page")

    address, text = messages.created[0]
    assert text == "debug-page"
    assert address.reference_message_id == "bot-message"
    assert address.owner_person_id == "admin"
    assert messages.tracked[0][1] == {
        "kind": OutboundMessageKind.COMMAND_REPLY,
        "state": OutboundMessageState.FINAL,
        "owner_person_id": "admin",
    }
