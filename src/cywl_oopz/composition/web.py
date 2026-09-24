"""Explicit construction of search, browser sessions, and curated tools."""

from __future__ import annotations

from dataclasses import dataclass

from cywl_oopz.features.agent.tools.ports import AgentTool
from cywl_oopz.features.agent.tools.web import (
    BrowserClickTool,
    BrowserCloseTool,
    BrowserFillTool,
    BrowserOpenTool,
    BrowserPressTool,
    BrowserSnapshotTool,
    BrowserWaitTool,
    ReadWebPageTool,
    SearchWebTool,
)
from cywl_oopz.features.web.browser import BrowserSessionManager
from cywl_oopz.features.web.service import WebSearchService
from cywl_oopz.integrations.web.agent_browser_mcp import AgentBrowserMcpGateway
from cywl_oopz.integrations.web.duckduckgo import DuckDuckGoSearchGateway
from cywl_oopz.settings import (
    AppSettings,
)


@dataclass(slots=True)
class WebComponents:
    web_search: WebSearchService | None = None
    browser: BrowserSessionManager | None = None
    tools: tuple[AgentTool, ...] = ()


def build_web(settings: AppSettings) -> WebComponents:
    """Build search/browser resources; BotApplication starts and closes them."""
    components = WebComponents()
    agent_tools: list[AgentTool] = []
    if settings.web.search_enabled:
        components.web_search = WebSearchService(
            settings.web,
            DuckDuckGoSearchGateway(
                timeout_seconds=settings.web.search_timeout_seconds,
                max_concurrency=settings.web.search_max_concurrency,
            ),
        )
        agent_tools.append(
            SearchWebTool(
                components.web_search,
                timeout_seconds=settings.agent.tool_timeout_seconds,
                max_output_characters=settings.agent.max_tool_result_characters,
            )
        )
    if settings.web.browser_enabled:
        components.browser = BrowserSessionManager(
            settings.web, AgentBrowserMcpGateway(settings.web)
        )
        browser_tool_options = {
            "timeout_seconds": max(
                settings.agent.tool_timeout_seconds,
                settings.web.browser_mcp_call_timeout_seconds + 2,
            ),
            "max_output_characters": settings.agent.max_tool_result_characters,
        }
        agent_tools.extend(
            (
                ReadWebPageTool(components.browser, **browser_tool_options),
                BrowserOpenTool(components.browser, **browser_tool_options),
                BrowserSnapshotTool(components.browser, **browser_tool_options),
                BrowserWaitTool(components.browser, **browser_tool_options),
                BrowserCloseTool(components.browser, **browser_tool_options),
            )
        )
        if settings.web.browser_interaction_enabled:
            agent_tools.extend(
                (
                    BrowserClickTool(components.browser, **browser_tool_options),
                    BrowserFillTool(components.browser, **browser_tool_options),
                    BrowserPressTool(components.browser, **browser_tool_options),
                )
            )
    components.tools = tuple(agent_tools)
    return components
