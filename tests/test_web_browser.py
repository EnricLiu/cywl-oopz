from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from cywl_oopz.features.agent.models import AgentIdentity, AgentRunLimits
from cywl_oopz.features.agent.tools.builtin import EmptyToolInput
from cywl_oopz.features.agent.tools.models import (
    ToolExecutionContext,
    ToolExecutionError,
)
from cywl_oopz.features.agent.tools.web import (
    BrowserClickTool,
    BrowserCloseTool,
    BrowserFillInput,
    BrowserFillTool,
    BrowserOpenTool,
    BrowserPressInput,
    BrowserPressTool,
    BrowserRefInput,
    BrowserSnapshotInput,
    BrowserSnapshotTool,
    BrowserWaitInput,
    BrowserWaitTool,
    ReadWebPageTool,
    WebPageUrlInput,
)
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.web.browser import BrowserSessionManager, PublicWebUrlPolicy
from cywl_oopz.features.web.errors import (
    BrowserContractError,
    BrowserNavigationError,
    BrowserStaleRefError,
    BrowserUnavailableError,
    WebPageUrlError,
)
from cywl_oopz.features.web.models import (
    BrowserActionResult,
    BrowserDocument,
    BrowserPageView,
    BrowserWaitKind,
    BrowserWaitRequest,
)
from cywl_oopz.integrations.web.agent_browser_mcp import AgentBrowserMcpGateway
from cywl_oopz.settings import WebToolsSettings

UPSTREAM_TOOLS = {
    "agent_browser_open": (),
    "agent_browser_read": (),
    "agent_browser_snapshot": (),
    "agent_browser_click": ("selector",),
    "agent_browser_fill": ("selector", "text"),
    "agent_browser_press": ("key",),
    "agent_browser_wait_ms": ("ms",),
    "agent_browser_wait_for_selector": ("selector",),
    "agent_browser_wait_for_text": ("text",),
    "agent_browser_wait_for_load": ("state",),
    "agent_browser_get_url": (),
    "agent_browser_get_title": (),
    "agent_browser_close": (),
}


def web_settings(**overrides: str) -> WebToolsSettings:
    return WebToolsSettings.from_mapping(
        {
            "CYWL_WEB_BROWSER_ENABLED": "true",
            **overrides,
        }
    )


def conversation(person: str, channel: str = "channel") -> ConversationKey:
    return ConversationKey("channel", "area", channel, person)


def tool_context() -> ToolExecutionContext:
    return ToolExecutionContext(
        run_id=uuid4(),
        identity=AgentIdentity("person", conversation("person")),
        limits=AgentRunLimits(),
        enabled_tools=(
            "read_web_page",
            "browser_open",
            "browser_snapshot",
            "browser_wait",
            "browser_close",
        ),
    )


class FakeBrowserGateway:
    def __init__(self) -> None:
        self.started = 0
        self.restarted = 0
        self.closed: list[str] = []
        self.sessions: list[str] = []
        self.urls: list[str] = []
        self.active = 0
        self.max_active = 0
        self.active_by_session: defaultdict[str, int] = defaultdict(int)
        self.max_active_by_session: defaultdict[str, int] = defaultdict(int)
        self.fail_once = False

    async def start(self) -> None:
        self.started += 1

    async def restart(self) -> None:
        self.restarted += 1

    async def _page(self, session: str, url: str) -> BrowserPageView:
        if self.fail_once:
            self.fail_once = False
            raise BrowserUnavailableError
        self.sessions.append(session)
        self.urls.append(url)
        self.active += 1
        self.active_by_session[session] += 1
        self.max_active = max(self.max_active, self.active)
        self.max_active_by_session[session] = max(
            self.max_active_by_session[session],
            self.active_by_session[session],
        )
        try:
            await asyncio.sleep(0.02)
        finally:
            self.active -= 1
            self.active_by_session[session] -= 1
        return BrowserPageView("Title", url, "snapshot", False)

    async def open(self, session: str, url: str) -> BrowserPageView:
        return await self._page(session, url)

    async def read(self, session: str, url: str | None) -> BrowserDocument:
        page = await self._page(session, url or "https://current.example")
        return BrowserDocument(page.title, page.url, "text/html", "content", False)

    async def snapshot(
        self,
        session: str,
        *,
        interactive: bool,
        compact: bool,
    ) -> BrowserPageView:
        del interactive, compact
        return await self._page(session, "https://current.example")

    async def wait(
        self,
        session: str,
        request: BrowserWaitRequest,
    ) -> BrowserPageView:
        del request
        return await self._page(session, "https://current.example")

    async def click(self, session: str, ref: str) -> BrowserPageView:
        del ref
        return await self._page(session, "https://clicked.example")

    async def fill(
        self,
        session: str,
        ref: str,
        text: str,
    ) -> BrowserActionResult:
        del ref, text
        self.sessions.append(session)
        return BrowserActionResult("Filled", "https://current.example", True)

    async def press(self, session: str, key: str) -> BrowserPageView:
        del key
        return await self._page(session, "https://pressed.example")

    async def close_session(self, session: str) -> None:
        self.closed.append(session)

    async def aclose(self) -> None:
        return None


class FakeMcpToolset:
    def __init__(
        self,
        *,
        missing: str | None = None,
        error_method: str | None = None,
        error_message: str = "navigation failed",
    ) -> None:
        self.server_info = SimpleNamespace(name="agent-browser", version="0.33.0")
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.entered = 0
        self.exited = 0
        self.missing = missing
        self.error_method = error_method
        self.error_message = error_message

    async def __aenter__(self) -> FakeMcpToolset:
        self.entered += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        del args
        self.exited += 1

    async def list_tools(self) -> list[SimpleNamespace]:
        tools = []
        for name, required in UPSTREAM_TOOLS.items():
            if name == self.missing:
                continue
            properties = {
                "session": {"type": "string"},
                "namespace": {"type": "string"},
                "timeoutMs": {"type": "integer"},
                **{property_name: {"type": "string"} for property_name in required},
            }
            tools.append(
                SimpleNamespace(
                    name=name,
                    inputSchema={
                        "additionalProperties": False,
                        "required": list(required),
                        "properties": properties,
                    },
                )
            )
        return tools

    async def direct_call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == self.error_method:
            raise ToolError(self.error_message)
        values: dict[str, dict[str, Any]] = {
            "agent_browser_open": {"title": "Opened", "url": "https://example.com/"},
            "agent_browser_read": {
                "content": "abcdefghijklmnopqrstuvwxyz",
                "contentType": "text/html",
                "finalUrl": "https://example.com/",
                "truncated": False,
            },
            "agent_browser_snapshot": {
                "snapshot": "0123456789abcdefghijklmnopqrstuvwxyz",
                "origin": "https://example.com/",
            },
            "agent_browser_get_url": {"url": "https://example.com/"},
            "agent_browser_get_title": {"title": "Example Domain"},
            "agent_browser_close": {"closed": True},
        }
        return {
            "exitCode": 0,
            "response": {
                "success": True,
                "data": values.get(name, {"waited": True}),
                "error": None,
            },
        }


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/data",
        "javascript:alert(1)",
        "https://user:password@example.com",
        "http://localhost/page",
        "http://service.localhost/page",
        "http://127.0.0.1/page",
        "http://10.0.0.1/page",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/page",
        "https://example.com/has space",
    ],
)
def test_public_web_url_policy_rejects_non_public_urls(url: str) -> None:
    with pytest.raises(WebPageUrlError):
        PublicWebUrlPolicy().validate(url)


def test_public_web_url_policy_accepts_ordinary_http_urls() -> None:
    policy = PublicWebUrlPolicy()

    assert policy.validate(" https://example.com/path?q=one ") == ("https://example.com/path?q=one")
    assert policy.validate("http://8.8.8.8/") == "http://8.8.8.8/"


def test_agent_browser_action_policy_defaults_to_deny() -> None:
    policy_path = (
        Path(__file__).parents[1]
        / "src"
        / "cywl_oopz"
        / "integrations"
        / "web"
        / "agent_browser_action_policy.json"
    )
    policy = json.loads(policy_path.read_text())

    assert policy["default"] == "deny"
    assert {"navigate", "read", "snapshot", "click", "fill", "press", "wait"}.issubset(
        policy["allow"]
    )
    assert {"url", "title", "close"}.issubset(policy["allow"])
    assert {
        "eval",
        "upload",
        "download",
        "auth",
        "cookies",
        "storage",
        "network",
        "screenshot",
    }.isdisjoint(policy["allow"])


@pytest.mark.asyncio
async def test_browser_sessions_are_opaque_isolated_and_serialized() -> None:
    gateway = FakeBrowserGateway()
    manager = BrowserSessionManager(web_settings(), gateway)
    first = conversation("person-one")
    second = conversation("person-two", "other")

    await asyncio.gather(
        manager.open(first, "https://example.com/one"),
        manager.open(first, "https://example.com/two"),
        manager.open(second, "https://example.org/three"),
    )

    first_name = manager.session_name(first)
    second_name = manager.session_name(second)
    assert first_name != second_name
    assert "person-one" not in first_name
    assert "channel" not in first_name
    assert len(first_name) <= 32
    assert gateway.max_active_by_session[first_name] == 1
    assert gateway.max_active == 2

    await manager.aclose()
    assert set(gateway.closed) == {first_name, second_name}


@pytest.mark.asyncio
async def test_browser_sessions_prune_idle_state_and_retry_one_broken_transport() -> None:
    now = 100.0
    gateway = FakeBrowserGateway()
    manager = BrowserSessionManager(
        web_settings(CYWL_WEB_BROWSER_SESSION_IDLE_SECONDS="10"),
        gateway,
        clock=lambda: now,
    )
    first = conversation("first")
    second = conversation("second")

    await manager.open(first, "https://example.com")
    now = 105.0
    gateway.fail_once = True
    await manager.open(first, "https://example.com/next")
    now = 116.0
    await manager.open(second, "https://example.org")

    assert manager.session_name(first) in gateway.closed
    assert gateway.restarted == 1
    assert gateway.urls[:3] == [
        "https://example.com",
        "https://example.com",
        "https://example.com/next",
    ]
    await manager.aclose()


@pytest.mark.asyncio
async def test_browser_writes_are_never_retried_after_transport_failure() -> None:
    gateway = FakeBrowserGateway()
    manager = BrowserSessionManager(web_settings(), gateway)
    key = conversation("writer")
    await manager.open(key, "https://example.com")
    gateway.fail_once = True

    with pytest.raises(BrowserUnavailableError):
        await manager.click(key, "@e1")

    assert gateway.restarted == 0
    await manager.aclose()


@pytest.mark.asyncio
async def test_agent_browser_gateway_validates_contract_and_bounds_outputs() -> None:
    toolset = FakeMcpToolset()
    gateway = AgentBrowserMcpGateway(
        web_settings(
            CYWL_WEB_BROWSER_MAX_CONTENT_CHARACTERS="20",
            CYWL_WEB_BROWSER_MAX_SNAPSHOT_CHARACTERS="24",
        ),
        toolset_factory=lambda: toolset,
    )

    await gateway.start()
    document = await gateway.read("session", "https://example.com")
    page = await gateway.open("session", "https://example.com")
    waited = await gateway.wait(
        "session",
        BrowserWaitRequest(BrowserWaitKind.TEXT, "ready", 2),
    )
    clicked = await gateway.click("session", "@e1")
    filled = await gateway.fill("session", "@e2", "hello")
    pressed = await gateway.press("session", "Enter")
    await gateway.close_session("session")
    await gateway.aclose()

    assert toolset.entered == 1
    assert toolset.exited == 1
    assert document.truncated is True
    assert len(document.content) == 20
    assert page.truncated is True
    assert waited.url == "https://example.com/"
    assert clicked.url == "https://example.com/"
    assert filled.applied is True
    assert pressed.url == "https://example.com/"
    assert all("extraArgs" not in arguments for _, arguments in toolset.calls)
    assert [name for name, _ in toolset.calls].count("agent_browser_snapshot") == 4


@pytest.mark.asyncio
async def test_agent_browser_gateway_rejects_contract_drift_and_maps_errors() -> None:
    incomplete = FakeMcpToolset(missing="agent_browser_read")
    invalid = AgentBrowserMcpGateway(
        web_settings(),
        toolset_factory=lambda: incomplete,
    )
    with pytest.raises(BrowserContractError):
        await invalid.start()
    assert incomplete.exited == 1

    failing_toolset = FakeMcpToolset(error_method="agent_browser_open")
    failing = AgentBrowserMcpGateway(
        web_settings(),
        toolset_factory=lambda: failing_toolset,
    )
    with pytest.raises(BrowserNavigationError):
        await failing.open("session", "https://example.com")
    await failing.aclose()

    stale_toolset = FakeMcpToolset(
        error_method="agent_browser_click",
        error_message="Element ref @e9 not found; stale ref",
    )
    stale = AgentBrowserMcpGateway(
        web_settings(),
        toolset_factory=lambda: stale_toolset,
    )
    with pytest.raises(BrowserStaleRefError):
        await stale.click("session", "@e9")
    await stale.aclose()


@pytest.mark.asyncio
async def test_browser_agent_tools_use_trusted_conversation_and_stable_errors() -> None:
    gateway = FakeBrowserGateway()
    manager = BrowserSessionManager(web_settings(), gateway)
    options = {"timeout_seconds": 5, "max_output_characters": 20_000}
    context = tool_context()

    document = await ReadWebPageTool(manager, **options).execute(
        context,
        WebPageUrlInput(url="https://example.com/article"),
    )
    opened = await BrowserOpenTool(manager, **options).execute(
        context,
        WebPageUrlInput(url="https://example.com/app"),
    )
    snapshot = await BrowserSnapshotTool(manager, **options).execute(
        context,
        BrowserSnapshotInput(),
    )
    waited = await BrowserWaitTool(manager, **options).execute(
        context,
        BrowserWaitInput(text="ready"),
    )
    clicked = await BrowserClickTool(manager, **options).execute(
        context,
        BrowserRefInput(ref="@e1"),
    )
    filled = await BrowserFillTool(manager, **options).execute(
        context,
        BrowserFillInput(ref="@e2", text="hello"),
    )
    pressed = await BrowserPressTool(manager, **options).execute(
        context,
        BrowserPressInput(key="Enter"),
    )
    closed = await BrowserCloseTool(manager, **options).execute(
        context,
        EmptyToolInput(),
    )

    assert document.url == "https://example.com/article"
    assert opened.snapshot == "snapshot"
    assert snapshot.url == "https://current.example"
    assert waited.title == "Title"
    assert clicked.url == "https://clicked.example"
    assert filled.applied is True
    assert pressed.url == "https://pressed.example"
    assert closed.closed is True
    assert set(gateway.sessions) == {manager.session_name(context.identity.conversation)}

    with pytest.raises(ToolExecutionError, match="web_page_url_invalid"):
        await ReadWebPageTool(manager, **options).execute(
            context,
            WebPageUrlInput(url="http://127.0.0.1/private"),
        )
    with pytest.raises(ValidationError, match="exactly one"):
        BrowserWaitInput(text="ready", selector="@e1")
    with pytest.raises(ValidationError):
        BrowserRefInput(ref="button.submit")
    with pytest.raises(ValidationError):
        BrowserPressInput(key="Control+Alt+Delete")
    await manager.aclose()


@pytest.mark.asyncio
async def test_real_agent_browser_gateway_when_explicitly_enabled() -> None:
    if os.getenv("CYWL_RUN_AGENT_BROWSER_MCP_TESTS") != "1":
        pytest.skip("set CYWL_RUN_AGENT_BROWSER_MCP_TESTS=1 for live browser adapter")
    gateway = AgentBrowserMcpGateway(web_settings(CYWL_WEB_BROWSER_DAEMON_IDLE_SECONDS="10"))
    session = "cywl-live-adapter"
    try:
        await gateway.start()
        page = await gateway.open(session, "https://example.com")
        document = await gateway.read(session, None)
        waited = await gateway.wait(
            session,
            BrowserWaitRequest(BrowserWaitKind.TEXT, "Example Domain", 5),
        )
        dynamic = await gateway.open(
            session,
            "https://www.selenium.dev/selenium/web/dynamic.html",
        )

        assert page.url == "https://example.com/"
        assert "Example Domain" in page.snapshot
        assert document.url == "https://example.com/"
        assert "Example Domain" in document.content
        assert "Example Domain" in waited.snapshot
        assert "Add a box!" in dynamic.snapshot
    finally:
        await gateway.close_session(session)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_real_agent_browser_interactions_when_explicitly_enabled() -> None:
    if os.getenv("CYWL_RUN_AGENT_BROWSER_MCP_TESTS") != "1":
        pytest.skip("set CYWL_RUN_AGENT_BROWSER_MCP_TESTS=1 for live interactions")
    gateway = AgentBrowserMcpGateway(
        web_settings(
            CYWL_WEB_BROWSER_INTERACTION_ENABLED="true",
            CYWL_WEB_BROWSER_DAEMON_IDLE_SECONDS="10",
        )
    )
    session = "cywl-live-actions"
    session_open = False
    try:
        await gateway.start()
        dynamic = await gateway.open(
            session,
            "https://www.selenium.dev/selenium/web/dynamic.html",
        )
        session_open = True
        assert "Add a box!" in dynamic.snapshot
        clicked = await gateway.click(session, "@e1")
        assert clicked.url.endswith("/selenium/web/dynamic.html")
        assert "Add a box!" in clicked.snapshot

        wikipedia = await gateway.open(session, "https://www.wikipedia.org")
        match = re.search(
            r'searchbox "Search Wikipedia".*?\[ref=(e\d+)\]',
            wikipedia.snapshot,
        )
        assert match is not None
        filled = await gateway.fill(session, f"@{match.group(1)}", "Hatsune Miku")
        assert filled.applied is True
        submitted = await gateway.press(session, "Enter")
        assert "wikipedia.org" in submitted.url
        assert "Hatsune" in submitted.url or "Hatsune" in submitted.snapshot
    finally:
        if session_open:
            await gateway.close_session(session)
        await gateway.aclose()


@pytest.mark.asyncio
async def test_real_action_policy_denies_eval_when_explicitly_enabled() -> None:
    if os.getenv("CYWL_RUN_AGENT_BROWSER_MCP_TESTS") != "1":
        pytest.skip("set CYWL_RUN_AGENT_BROWSER_MCP_TESTS=1 for live action policy")
    executable = shutil.which("agent-browser")
    assert executable is not None
    policy_path = (
        Path(__file__).parents[1]
        / "src"
        / "cywl_oopz"
        / "integrations"
        / "web"
        / "agent_browser_action_policy.json"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "AGENT_BROWSER_SESSION": "cywl-policy-test",
            "AGENT_BROWSER_NAMESPACE": "cywl-oopz-policy-test",
            "AGENT_BROWSER_ACTION_POLICY": str(policy_path),
            "AGENT_BROWSER_IDLE_TIMEOUT_MS": "2000",
        }
    )

    async def command(*arguments: str) -> tuple[int, bytes]:
        process = await asyncio.create_subprocess_exec(
            executable,
            *arguments,
            "--json",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(process.communicate(), timeout=15)
        return process.returncode or 0, output

    opened_code, _ = await command("open", "https://example.com")
    eval_code, eval_output = await command("eval", "1 + 1")
    closed_code, _ = await command("close")

    assert opened_code == 0
    assert eval_code != 0
    assert b"denied by policy" in eval_output
    assert closed_code == 0
