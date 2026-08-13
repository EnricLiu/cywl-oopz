from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest

from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.features.access.models import AccessRole, RoleBinding, RoleBindingScope
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.features.admin.commands import DebugCommand
from cywl_oopz.features.admin.models import (
    AgentDiagnosticTool,
    AgentResponseDiagnostic,
    OopzMessageAddress,
    OopzMessageScope,
    OutboundMessageKind,
    OutboundMessageReceipt,
    OutboundMessageState,
)
from cywl_oopz.integrations.oopz.diagnostic_renderer import OopzAgentDiagnosticRenderer
from cywl_oopz.integrations.oopz.message_renderer import oopz_units
from cywl_oopz.integrations.oopz.tracked_context import TrackedMessageContext

RUN_ID = UUID("10000000-0000-0000-0000-000000000001")
THREAD_ID = UUID("10000000-0000-0000-0000-000000000002")


@dataclass
class RecordingReceipts:
    records: list[OutboundMessageReceipt] = field(default_factory=list)

    async def create(self, receipt: OutboundMessageReceipt) -> bool:
        self.records.append(receipt)
        return True


class FakeSdkContext:
    def __init__(self, *, private: bool = False) -> None:
        self.event = SimpleNamespace(
            is_private=private,
            message=SimpleNamespace(
                message_id="source",
                sender_id="person",
                area="" if private else "area",
                channel="channel",
            ),
        )
        self.bot = SimpleNamespace()
        self.config = SimpleNamespace()

    async def reply(self, *text: str, **kwargs):
        del text, kwargs
        return SimpleNamespace(message_id="reply", timestamp="123")

    async def send(self, *text: str, **kwargs):
        del text, kwargs
        return SimpleNamespace(message_id="send", timestamp="124")


@pytest.mark.asyncio
@pytest.mark.parametrize("private", [False, True])
async def test_tracked_context_records_successful_reply_and_send(private: bool) -> None:
    repository = RecordingReceipts()
    context = TrackedMessageContext(FakeSdkContext(private=private), repository)

    await context.reply("reply")
    await context.send("send")

    assert [record.message_id for record in repository.records] == ["reply", "send"]
    assert repository.records[0].in_reply_to_message_id == "source"
    assert repository.records[1].in_reply_to_message_id == ""
    assert repository.records[0].address.scope is (
        OopzMessageScope.PRIVATE if private else OopzMessageScope.CHANNEL
    )


def diagnostic(*, answer: str = "完整答案", secret: str = "top-secret") -> AgentResponseDiagnostic:
    now = datetime.now(UTC)
    receipt = OutboundMessageReceipt(
        "agent-message",
        "123",
        OutboundMessageKind.AGENT_RESPONSE,
        OutboundMessageState.FINAL,
        OopzMessageAddress(OopzMessageScope.CHANNEL, "area", "channel"),
        in_reply_to_message_id="source",
        owner_person_id="person",
        agent_run_id=RUN_ID,
        diagnostic_snapshot={
            "phase": "succeeded",
            "final_text": answer,
            "elapsed_seconds": 8.4,
            "input_tokens": 1200,
            "output_tokens": 384,
            "model_requests": 2,
            "tool_calls": 1,
            "provider_retry_count": 1,
            "provider_retries": [
                {"attempt": 1, "max_attempts": 2, "delay_seconds": 0.5, "reason": "HTTP 429"}
            ],
        },
    )
    tool = AgentDiagnosticTool(
        "call-1",
        "search_web",
        "1",
        "read",
        "succeeded",
        {"query": "初音未来 新闻", "api_key": secret},
        {
            "results": [{"url": "https://example.com"}],
            "media_url": "https://media.example/audio?sign=media-secret&expires=123",
            "token": secret,
        },
        "",
        now,
        now + timedelta(milliseconds=800),
    )
    return AgentResponseDiagnostic(
        receipt,
        RUN_ID,
        THREAD_ID,
        "succeeded",
        "completed",
        provider_alias="openrouter",
        model_alias="qwen",
        selection_source="user",
        limits={"max_tool_calls": 8},
        usage={"input_tokens": 1200, "output_tokens": 384, "tool_calls": 1},
        assistant_text=answer,
        started_at=now,
        finished_at=now + timedelta(seconds=8.4),
        tools=(tool,),
    )


def test_diagnostic_renderer_restores_full_answer_and_redacts_verbose_payloads() -> None:
    renderer = OopzAgentDiagnosticRenderer({"search_web": "搜索公开网页"})

    pages = renderer.render(diagnostic(), verbose=True)
    rendered = "\n".join(pages)

    assert "完整答案" in rendered
    assert "搜索公开网页" in rendered
    assert "openrouter/qwen" in rendered
    assert "1200→384 tokens" in rendered
    assert "top-secret" not in rendered
    assert "media-secret" not in rendered
    assert "https://media.example/audio" in rendered
    assert "[已隐藏]" in rendered
    assert all(oopz_units(page) <= 1950 for page in pages)
    assert len(pages) <= 8


def test_diagnostic_renderer_bounds_a_very_long_answer() -> None:
    renderer = OopzAgentDiagnosticRenderer()

    pages = renderer.render(diagnostic(answer="长答案\n\n" * 10_000), verbose=False)

    assert len(pages) == 8
    assert all(oopz_units(page) <= 1950 for page in pages)
    assert "其余诊断内容已截断" in pages[-1]


def test_diagnostic_renderer_handles_running_and_expired_runs() -> None:
    renderer = OopzAgentDiagnosticRenderer()
    now = datetime.now(UTC)
    running_receipt = replace(
        diagnostic().receipt,
        state=OutboundMessageState.ACTIVE,
        diagnostic_snapshot={"phase": "running", "input_tokens": None},
    )
    running = replace(
        diagnostic(),
        receipt=running_receipt,
        status="running",
        assistant_text="",
        started_at=now - timedelta(seconds=2),
        finished_at=None,
    )
    expired_receipt = replace(
        diagnostic(answer="快照保留的完整回答").receipt,
        diagnostic_snapshot={
            "phase": "succeeded",
            "final_text": "快照保留的完整回答",
            "input_tokens": 10,
            "output_tokens": 5,
        },
    )
    expired = AgentResponseDiagnostic(receipt=expired_receipt, run_id=RUN_ID)

    running_text = "\n".join(renderer.render(running, verbose=False))
    expired_text = "\n".join(renderer.render(expired, verbose=False))

    assert "仍在运行" in running_text
    assert "回复仍在运行" in running_text
    assert "1200→384 tokens" in running_text
    assert "快照保留的完整回答" in expired_text


@dataclass
class RoleRepository:
    records: tuple[RoleBinding, ...]

    async def list_for_subject(self, person_id: str) -> tuple[RoleBinding, ...]:
        return tuple(record for record in self.records if record.subject_person_id == person_id)


class DiagnosticRepository:
    def __init__(self, value: AgentResponseDiagnostic | None) -> None:
        self.value = value
        self.calls: list[tuple[str, OopzMessageAddress]] = []

    async def get_by_outbound_message(self, message_id, address):
        self.calls.append((message_id, address))
        return self.value


class FakeMessage:
    def __init__(self, text: str, *, reference: str = "agent-message") -> None:
        self.plain_text = text
        self.text = text
        self.content = text
        self.sender_id = "admin"
        self.area = "area"
        self.channel = "channel"
        self.reference_message_id = reference


class FakeCommandContext:
    def __init__(self, message: FakeMessage) -> None:
        self.event = SimpleNamespace(message=message, is_private=False)
        self.replies: list[str] = []

    async def reply(self, text: str) -> None:
        self.replies.append(text)


@pytest.mark.asyncio
async def test_debug_requires_permission_and_exact_reference_address() -> None:
    roles = RoleRepository(
        (
            RoleBinding(
                "admin",
                AccessRole.ADMIN,
                RoleBindingScope.AREA,
                area_id="area",
            ),
        )
    )
    repository = DiagnosticRepository(diagnostic())
    router = CommandRouter("/", AuthorizationService(roles))
    router.register_definition(DebugCommand(repository, OopzAgentDiagnosticRenderer()).definition())
    message = FakeMessage("/debug --verbose")
    context = FakeCommandContext(message)

    assert await router.dispatch(message, context)

    assert context.replies
    assert repository.calls == [
        (
            "agent-message",
            OopzMessageAddress(OopzMessageScope.CHANNEL, "area", "channel"),
        )
    ]


@pytest.mark.asyncio
async def test_debug_rejects_unprivileged_user_before_reading_diagnostic() -> None:
    repository = DiagnosticRepository(diagnostic())
    router = CommandRouter("/", AuthorizationService(RoleRepository(())))
    router.register_definition(DebugCommand(repository, OopzAgentDiagnosticRenderer()).definition())
    message = FakeMessage("/debug")
    context = FakeCommandContext(message)

    assert await router.dispatch(message, context)

    assert context.replies == ["你没有执行此操作的权限。"]
    assert repository.calls == []
