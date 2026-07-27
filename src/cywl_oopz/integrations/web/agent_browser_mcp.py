"""Project-owned high-level adapter for the agent-browser stdio MCP server."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Callable, Mapping
from typing import Any

from fastmcp.exceptions import ToolError
from pydantic_ai.mcp import MCPToolset, StdioTransport

from cywl_oopz.features.web.errors import (
    BrowserActionError,
    BrowserContractError,
    BrowserError,
    BrowserNavigationError,
    BrowserStaleRefError,
    BrowserTimeoutError,
    BrowserUnavailableError,
)
from cywl_oopz.features.web.models import (
    BrowserDocument,
    BrowserPageView,
    BrowserWaitKind,
    BrowserWaitRequest,
)
from cywl_oopz.settings import WebToolsSettings

logger = logging.getLogger(__name__)

_SERVER_NAME = "agent-browser"
_SERVER_VERSION = "0.33.0"
_NAMESPACE = "cywl-oopz"
_REQUIRED_SCHEMAS: Mapping[str, tuple[str, ...]] = {
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


class AgentBrowserMcpGateway:
    """Expose only fixed, typed browser use cases over a validated MCP contract."""

    def __init__(
        self,
        settings: WebToolsSettings,
        *,
        toolset_factory: Callable[[], MCPToolset] | None = None,
    ) -> None:
        self._settings = settings
        self._toolset_factory = toolset_factory or self._build_toolset
        self._requires_executable = toolset_factory is None
        self._toolset: MCPToolset | None = None
        self._lifecycle_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the stdio transport and reject incompatible upstream schemas."""
        async with self._lifecycle_lock:
            if self._toolset is not None:
                return
            await self._start_unlocked()

    async def restart(self) -> None:
        """Close a suspect transport and initialize one replacement."""
        async with self._lifecycle_lock:
            await self._close_unlocked()
            await self._start_unlocked()

    async def _start_unlocked(self) -> None:
        if self._requires_executable and shutil.which("agent-browser") is None:
            raise BrowserUnavailableError
        toolset = self._toolset_factory()
        entered = False
        try:
            await asyncio.wait_for(
                toolset.__aenter__(),
                timeout=self._settings.browser_mcp_init_timeout_seconds,
            )
            entered = True
            await self._validate_contract(toolset)
        except asyncio.CancelledError:
            if entered:
                await toolset.__aexit__(None, None, None)
            raise
        except TimeoutError as exc:
            if entered:
                await toolset.__aexit__(None, None, None)
            raise BrowserTimeoutError from exc
        except BrowserError:
            if entered:
                await toolset.__aexit__(None, None, None)
            raise
        except Exception as exc:
            if entered:
                await toolset.__aexit__(None, None, None)
            logger.debug("agent-browser MCP initialization failed", exc_info=exc)
            raise BrowserUnavailableError from exc
        self._toolset = toolset

    async def _validate_contract(self, toolset: MCPToolset) -> None:
        server_info = toolset.server_info
        if (
            server_info is None
            or server_info.name != _SERVER_NAME
            or server_info.version != _SERVER_VERSION
        ):
            raise BrowserContractError
        discovered = {tool.name: tool for tool in await toolset.list_tools()}
        for name, required in _REQUIRED_SCHEMAS.items():
            tool = discovered.get(name)
            if tool is None:
                raise BrowserContractError
            schema = tool.inputSchema
            properties = schema.get("properties", {})
            if (
                schema.get("additionalProperties") is not False
                or tuple(schema.get("required", ())) != required
                or any(property_name not in properties for property_name in required)
                or "session" not in properties
                or "namespace" not in properties
                or "timeoutMs" not in properties
            ):
                raise BrowserContractError

    def _build_toolset(self) -> MCPToolset:
        environment = dict(os.environ)
        environment.update(
            {
                "AGENT_BROWSER_NAMESPACE": _NAMESPACE,
                "AGENT_BROWSER_CONTENT_BOUNDARIES": "true",
                "AGENT_BROWSER_MAX_OUTPUT": str(
                    max(
                        self._settings.browser_max_content_characters,
                        self._settings.browser_max_snapshot_characters,
                    )
                    + 2_000
                ),
                "AGENT_BROWSER_IDLE_TIMEOUT_MS": str(
                    self._settings.browser_daemon_idle_seconds * 1_000
                ),
                "AGENT_BROWSER_NO_AUTO_DIALOG": "false",
            }
        )
        transport = StdioTransport(
            "agent-browser",
            ["mcp", "--tools", "core"],
            env=environment,
            cwd=os.getcwd(),
        )
        return MCPToolset(
            transport,
            tool_error_behavior="error",
            include_instructions=False,
            cache_tools=True,
            init_timeout=self._settings.browser_mcp_init_timeout_seconds,
            read_timeout=self._settings.browser_mcp_call_timeout_seconds,
        )

    async def open(self, session: str, url: str) -> BrowserPageView:
        """Navigate and return a fresh compact interactive snapshot."""
        await self._call(
            "agent_browser_open",
            {
                "url": url,
                "headed": False,
                **self._common(session),
            },
        )
        return await self.snapshot(session, interactive=True, compact=True)

    async def read(self, session: str, url: str | None) -> BrowserDocument:
        """Read HTML/Markdown content without requiring a Markdown content type."""
        arguments: dict[str, Any] = self._common(session)
        if url is not None:
            arguments["url"] = url
        data = await self._call("agent_browser_read", arguments)
        content, truncated_here = self._bounded_text(
            str(data.get("content", "")),
            self._settings.browser_max_content_characters,
        )
        final_url = str(data.get("finalUrl") or data.get("url") or url or "")
        try:
            title_data = await self._call(
                "agent_browser_get_title",
                self._common(session),
            )
            title = str(title_data.get("title", "")).strip()
        except BrowserError:
            title = ""
        return BrowserDocument(
            title=title or final_url,
            url=final_url,
            content_type=str(data.get("contentType", "")),
            content=content,
            truncated=bool(data.get("truncated", False)) or truncated_here,
        )

    async def snapshot(
        self,
        session: str,
        *,
        interactive: bool,
        compact: bool,
    ) -> BrowserPageView:
        """Return bounded snapshot text plus provider-confirmed URL and title."""
        snapshot_data = await self._call(
            "agent_browser_snapshot",
            {
                "interactive": interactive,
                "compact": compact,
                "includeUrls": True,
                **self._common(session),
            },
        )
        url_data = await self._call(
            "agent_browser_get_url",
            self._common(session),
        )
        title_data = await self._call(
            "agent_browser_get_title",
            self._common(session),
        )
        snapshot, truncated = self._bounded_text(
            str(snapshot_data.get("snapshot", "")),
            self._settings.browser_max_snapshot_characters,
        )
        return BrowserPageView(
            title=str(title_data.get("title", "")).strip(),
            url=str(
                url_data.get("url") or snapshot_data.get("origin") or snapshot_data.get("url") or ""
            ),
            snapshot=snapshot,
            truncated=truncated,
        )

    async def wait(
        self,
        session: str,
        request: BrowserWaitRequest,
    ) -> BrowserPageView:
        """Map one project wait condition to a fixed upstream method."""
        timeout_ms = max(int(request.timeout_seconds * 1_000), 1)
        arguments: dict[str, Any] = self._common(session)
        if request.kind is BrowserWaitKind.LOAD:
            method = "agent_browser_wait_for_load"
            arguments.update(state=str(request.value), waitTimeoutMs=timeout_ms)
        elif request.kind is BrowserWaitKind.TEXT:
            method = "agent_browser_wait_for_text"
            arguments.update(text=str(request.value), waitTimeoutMs=timeout_ms)
        elif request.kind is BrowserWaitKind.SELECTOR:
            method = "agent_browser_wait_for_selector"
            arguments.update(selector=str(request.value), waitTimeoutMs=timeout_ms)
        else:
            method = "agent_browser_wait_ms"
            arguments.update(ms=int(request.value))
        await self._call(method, arguments)
        return await self.snapshot(session, interactive=True, compact=True)

    async def close_session(self, session: str) -> None:
        """Close only the selected project-owned session, never all sessions."""
        await self._call(
            "agent_browser_close",
            {"all": False, **self._common(session)},
        )

    def _common(self, session: str) -> dict[str, Any]:
        return {
            "session": session,
            "namespace": _NAMESPACE,
            "timeoutMs": max(
                int(self._settings.browser_mcp_call_timeout_seconds * 1_000),
                1,
            ),
        }

    async def _call(self, method: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if method not in _REQUIRED_SCHEMAS:
            raise BrowserContractError
        if self._toolset is None:
            await self.start()
        toolset = self._toolset
        if toolset is None:
            raise BrowserUnavailableError
        try:
            async with asyncio.timeout(self._settings.browser_mcp_call_timeout_seconds + 1):
                result = await toolset.direct_call_tool(method, arguments)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise BrowserTimeoutError from exc
        except ToolError as exc:
            raise self._mapped_tool_error(method, str(exc)) from exc
        except Exception as exc:
            logger.debug("agent-browser MCP call failed: %s", method, exc_info=exc)
            raise BrowserUnavailableError from exc
        if not isinstance(result, dict):
            raise BrowserUnavailableError
        response = result.get("response")
        if not isinstance(response, dict) or response.get("success") is not True:
            error = str(response.get("error", "")) if isinstance(response, dict) else ""
            raise self._mapped_tool_error(method, error)
        data = response.get("data")
        if not isinstance(data, dict):
            raise BrowserUnavailableError
        return data

    @staticmethod
    def _mapped_tool_error(method: str, message: str) -> BrowserError:
        normalized = message.casefold()
        if "timed out" in normalized or "timeout" in normalized:
            return BrowserTimeoutError()
        if "failed to connect" in normalized or "no such file" in normalized:
            return BrowserUnavailableError()
        if "ref" in normalized and (
            "not found" in normalized or "stale" in normalized or "no element" in normalized
        ):
            return BrowserStaleRefError()
        if method in {"agent_browser_open", "agent_browser_read"}:
            return BrowserNavigationError()
        return BrowserActionError()

    @staticmethod
    def _bounded_text(value: str, limit: int) -> tuple[str, bool]:
        if len(value) <= limit:
            return value, False
        marker = "\n…（网页内容已截断）…\n"
        if limit <= len(marker):
            return value[:limit], True
        available = max(limit - len(marker), 0)
        leading = int(available * 0.75)
        trailing = available - leading
        return f"{value[:leading]}{marker}{value[-trailing:] if trailing else ''}", True

    async def _close_unlocked(self) -> None:
        toolset = self._toolset
        self._toolset = None
        if toolset is not None:
            await toolset.__aexit__(None, None, None)

    async def aclose(self) -> None:
        """Close only the MCP stdio transport owned by this adapter."""
        async with self._lifecycle_lock:
            await self._close_unlocked()
