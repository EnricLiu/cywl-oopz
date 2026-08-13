from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    CommandDefinition,
    CommandExecutionPolicy,
    ExecutionMode,
    NoArguments,
    NoArgumentsParser,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.execution import CommandTaskSupervisor
from cywl_oopz.commands.models import (
    CommandActor,
    CommandLocation,
    CommandRequest,
    CommandScope,
    CommandSource,
    CommandText,
    CommandTrigger,
    DispatchStatus,
)
from cywl_oopz.commands.router import CommandRouter


@dataclass
class RecordingResponder:
    replies: list[str] = field(default_factory=list)

    async def reply(self, message: str) -> None:
        self.replies.append(message)

    async def send(self, message: str) -> None:
        self.replies.append(message)

    async def react(self, emoji: str) -> None:
        del emoji


def command_request(name: str) -> tuple[CommandRequest, RecordingResponder]:
    responder = RecordingResponder()
    return (
        CommandRequest(
            CommandTrigger.TEXT,
            CommandActor("actor"),
            CommandLocation(CommandScope.CHANNEL, "area", "channel"),
            CommandSource(f"message-{name}"),
            responder,
            CommandText(f"/{name}", name, "", ()),
        ),
        responder,
    )


class BlockingHandler:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del request, arguments
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class RaisingHandler:
    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del request, arguments
        raise RuntimeError("secret internal failure")


def definition(
    name: str,
    handler: object,
    *,
    mode: ExecutionMode = ExecutionMode.BACKGROUND,
    timeout_seconds: float | None = None,
) -> CommandDefinition[NoArguments]:
    return CommandDefinition(
        CommandSpec(name, f"{name} command", "test", (name,)),
        NoArgumentsParser(),
        handler,  # type: ignore[arg-type]
        PublicCommandAuthorization(),
        CommandExecutionPolicy(mode, timeout_seconds),
    )


async def wait_for_idle(supervisor: CommandTaskSupervisor) -> None:
    for _ in range(100):
        if supervisor.active_count == 0:
            return
        await asyncio.sleep(0)
    raise AssertionError("command supervisor did not become idle")


@pytest.mark.asyncio
async def test_background_command_returns_started_without_blocking_next_dispatch() -> None:
    supervisor = CommandTaskSupervisor(drain_timeout_seconds=0)
    handler = BlockingHandler()
    router = CommandRouter("/", supervisor=supervisor)
    router.register_definition(definition("slow", handler))

    class FastHandler:
        async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
            del arguments
            await request.responder.reply("fast complete")

    router.register_definition(definition("fast", FastHandler(), mode=ExecutionMode.INLINE))
    slow_request, _ = command_request("slow")
    fast_request, fast_responder = command_request("fast")

    slow_outcome = await router.dispatch_request(slow_request)
    fast_outcome = await router.dispatch_request(fast_request)

    assert slow_outcome.status is DispatchStatus.STARTED
    assert fast_outcome.status is DispatchStatus.COMPLETED
    assert fast_responder.replies == ["fast complete"]
    await handler.started.wait()
    handler.release.set()
    await wait_for_idle(supervisor)
    await supervisor.close()


@pytest.mark.asyncio
async def test_command_timeout_is_mapped_to_one_safe_failure() -> None:
    handler = BlockingHandler()
    router = CommandRouter("/")
    router.register_definition(
        definition(
            "slow",
            handler,
            mode=ExecutionMode.INLINE,
            timeout_seconds=0.01,
        )
    )
    request, responder = command_request("slow")

    outcome = await router.dispatch_request(request)

    assert outcome.status is DispatchStatus.FAILED
    assert responder.replies == ["命令执行超时，请稍后重试。"]
    assert handler.cancelled is True


@pytest.mark.asyncio
async def test_unexpected_handler_error_is_consumed_and_not_exposed() -> None:
    router = CommandRouter("/")
    router.register_definition(definition("broken", RaisingHandler(), mode=ExecutionMode.INLINE))
    request, responder = command_request("broken")

    outcome = await router.dispatch_request(request)

    assert outcome.status is DispatchStatus.FAILED
    assert responder.replies == ["命令执行失败，请稍后重试。"]
    assert "secret" not in responder.replies[0]


@pytest.mark.asyncio
async def test_supervisor_close_cancels_and_awaits_remaining_commands() -> None:
    supervisor = CommandTaskSupervisor(drain_timeout_seconds=0)
    handler = BlockingHandler()
    request, _ = command_request("slow")
    operation = handler.handle(request, NoArguments())

    assert supervisor.start("slow", "request", operation) is True
    await handler.started.wait()
    await supervisor.close()

    assert handler.cancelled is True
    assert supervisor.active_count == 0
    assert supervisor.accepting is False


@pytest.mark.asyncio
async def test_supervisor_drains_completed_work_and_rejects_new_work() -> None:
    supervisor = CommandTaskSupervisor(drain_timeout_seconds=0.1)
    completed = asyncio.Event()

    async def finish() -> None:
        completed.set()

    assert supervisor.start("short", "request", finish()) is True
    await supervisor.close()
    assert completed.is_set()

    rejected_ran = False

    async def rejected() -> None:
        nonlocal rejected_ran
        rejected_ran = True

    assert supervisor.start("late", "request", rejected()) is False
    assert rejected_ran is False
