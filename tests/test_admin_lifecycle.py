from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import cywl_oopz.__main__ as main_module
from cywl_oopz.application import BotApplication
from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.features.access.models import AccessRole, RoleBinding, RoleBindingScope
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.features.admin.commands import RebootCommand
from cywl_oopz.features.admin.lifecycle import ApplicationLifecycleCoordinator
from cywl_oopz.features.admin.models import ShutdownDisposition
from cywl_oopz.testing.commands import dispatch_command


@pytest.mark.asyncio
async def test_lifecycle_restart_is_first_writer_wins_after_confirmation() -> None:
    lifecycle = ApplicationLifecycleCoordinator()
    confirmations: list[str] = []

    async def first_confirmation() -> None:
        confirmations.append("first")
        await asyncio.sleep(0)

    async def second_confirmation() -> None:
        confirmations.append("second")

    accepted = await asyncio.gather(
        lifecycle.request_restart("actor-1", first_confirmation),
        lifecycle.request_restart("actor-2", second_confirmation),
    )

    assert accepted == [True, False]
    assert confirmations == ["first"]
    assert lifecycle.restart_requested is True
    assert await lifecycle.wait() is ShutdownDisposition.RESTART


@pytest.mark.asyncio
async def test_failed_restart_confirmation_does_not_commit_shutdown() -> None:
    lifecycle = ApplicationLifecycleCoordinator()

    async def failure() -> None:
        raise RuntimeError("reply failed")

    with pytest.raises(RuntimeError, match="reply failed"):
        await lifecycle.request_restart("actor", failure)

    assert lifecycle.restart_requested is False
    assert lifecycle.disposition is ShutdownDisposition.NORMAL

    async def success() -> None:
        return None

    assert await lifecycle.request_restart("actor", success) is True


@dataclass
class RoleRepository:
    records: tuple[RoleBinding, ...]

    async def list_for_subject(self, person_id: str) -> tuple[RoleBinding, ...]:
        return tuple(item for item in self.records if item.subject_person_id == person_id)


class CommandMessage:
    def __init__(self, text: str = "/reboot", *, sender: str = "admin") -> None:
        self.plain_text = text
        self.text = text
        self.content = text
        self.sender_id = sender
        self.area = "area"
        self.channel = "channel"


class CommandContext:
    def __init__(self, message: CommandMessage) -> None:
        self.event = SimpleNamespace(message=message, is_private=False)
        self.replies: list[str] = []

    async def reply(self, text: str) -> object:
        self.replies.append(text)
        return SimpleNamespace(message_id=f"reply-{len(self.replies)}", timestamp="123")


def reboot_router(
    lifecycle: ApplicationLifecycleCoordinator,
    binding: RoleBinding,
) -> CommandRouter:
    router = CommandRouter("/", AuthorizationService(RoleRepository((binding,))))
    router.register_definition(RebootCommand(lifecycle).definition())
    return router


@pytest.mark.asyncio
async def test_reboot_requires_global_permission_and_merges_duplicates() -> None:
    global_admin = RoleBinding("admin", AccessRole.ADMIN, RoleBindingScope.GLOBAL)
    lifecycle = ApplicationLifecycleCoordinator()
    router = reboot_router(lifecycle, global_admin)
    first_message = CommandMessage()
    first = CommandContext(first_message)
    second_message = CommandMessage()
    second = CommandContext(second_message)

    assert await dispatch_command(router, first_message, first)
    assert await dispatch_command(router, second_message, second)

    assert first.replies == ["🔄 **正在重启…**"]
    assert second.replies == ["重启已经在进行中。"]

    area_admin = RoleBinding(
        "admin",
        AccessRole.ADMIN,
        RoleBindingScope.AREA,
        area_id="area",
    )
    denied_lifecycle = ApplicationLifecycleCoordinator()
    denied_router = reboot_router(denied_lifecycle, area_admin)
    denied_message = CommandMessage()
    denied = CommandContext(denied_message)
    await dispatch_command(denied_router, denied_message, denied)

    assert denied.replies == ["你没有执行此操作的权限。"]
    assert denied_lifecycle.restart_requested is False


@pytest.mark.asyncio
async def test_reboot_rejects_arguments_before_requesting_restart() -> None:
    lifecycle = ApplicationLifecycleCoordinator()
    router = reboot_router(
        lifecycle,
        RoleBinding("admin", AccessRole.ADMIN, RoleBindingScope.GLOBAL),
    )
    message = CommandMessage("/reboot now")
    context = CommandContext(message)

    await dispatch_command(router, message, context)

    assert context.replies == ["此命令不接受额外参数。\n用法：/reboot"]
    assert lifecycle.restart_requested is False


@pytest.mark.asyncio
async def test_oopz_stop_timeout_still_releases_the_run_task() -> None:
    run_released = asyncio.Event()
    stop_released = asyncio.Event()

    class StuckBot:
        async def run(self) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                run_released.set()

        async def stop(self) -> None:
            try:
                await asyncio.Event().wait()
            finally:
                stop_released.set()

    application = object.__new__(BotApplication)
    application.bot = StuckBot()
    application.lifecycle = ApplicationLifecycleCoordinator()
    application.oopz_stop_timeout_seconds = 0.01
    application.oopz_run_settle_seconds = 0.01

    async def confirmed() -> None:
        return None

    assert await application.lifecycle.request_restart("actor", confirmed)
    disposition = await application._run_oopz_until_lifecycle_request()

    assert disposition is ShutdownDisposition.RESTART
    assert stop_released.is_set()
    assert run_released.is_set()


def test_cli_uses_exit_code_75_only_for_restart(monkeypatch) -> None:
    class FakeSettings:
        @staticmethod
        def from_environment() -> object:
            return SimpleNamespace(
                agent=SimpleNamespace(enabled=False, live_display=False),
                music=SimpleNamespace(enabled=False),
                web=SimpleNamespace(search_enabled=False, browser_enabled=False),
            )

    class RestartingApplication:
        def __init__(self, settings: object) -> None:
            del settings

        async def run(self) -> ShutdownDisposition:
            return ShutdownDisposition.RESTART

    monkeypatch.setattr(main_module, "AppSettings", FakeSettings)
    monkeypatch.setattr(main_module, "BotApplication", RestartingApplication)

    with pytest.raises(SystemExit) as exit_info:
        main_module.main()

    assert exit_info.value.code == 75

    class NormalApplication(RestartingApplication):
        async def run(self) -> ShutdownDisposition:
            return ShutdownDisposition.NORMAL

    monkeypatch.setattr(main_module, "BotApplication", NormalApplication)
    main_module.main()
