from __future__ import annotations

import asyncio
import os
import threading
from typing import Any
from uuid import uuid4

import pytest
from ddgs.exceptions import RatelimitException, TimeoutException

from cywl_oopz.features.agent.models import AgentIdentity, AgentRunLimits
from cywl_oopz.features.agent.tools.models import ToolExecutionContext, ToolExecutionError
from cywl_oopz.features.agent.tools.web import SearchWebInput, SearchWebTool
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.web.errors import (
    WebSearchQueryError,
    WebSearchRateLimitError,
    WebSearchTimeoutError,
)
from cywl_oopz.features.web.models import (
    WebSearchRequest,
    WebSearchResult,
    WebSearchTimeRange,
)
from cywl_oopz.features.web.service import WebSearchService
from cywl_oopz.integrations.web.duckduckgo import DuckDuckGoSearchGateway
from cywl_oopz.settings import WebSearchSafeSearch, WebToolsSettings


class RecordingDdgs:
    def __init__(
        self,
        *,
        results: list[dict[str, str]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.thread_id: int | None = None
        self.arguments: dict[str, Any] = {}

    def text(self, query: str, **kwargs: Any) -> list[dict[str, str]]:
        self.thread_id = threading.get_ident()
        self.arguments = {"query": query, **kwargs}
        if self.error is not None:
            raise self.error
        return self.results


class StaticGateway:
    def __init__(self, results: tuple[WebSearchResult, ...] = ()) -> None:
        self.results = results
        self.request: WebSearchRequest | None = None

    async def search(self, request: WebSearchRequest) -> tuple[WebSearchResult, ...]:
        self.request = request
        return self.results

    async def aclose(self) -> None:
        return None


def web_settings(**overrides: str) -> WebToolsSettings:
    return WebToolsSettings.from_mapping(overrides)


def tool_context() -> ToolExecutionContext:
    conversation = ConversationKey("channel", "area", "channel", "person")
    return ToolExecutionContext(
        run_id=uuid4(),
        identity=AgentIdentity("person", conversation),
        limits=AgentRunLimits(),
        enabled_tools=("search_web",),
    )


@pytest.mark.asyncio
async def test_duckduckgo_gateway_runs_off_loop_and_normalizes_results() -> None:
    main_thread_id = threading.get_ident()
    client = RecordingDdgs(
        results=[
            {
                "title": "  Example   result  ",
                "href": "https://example.com/page",
                "body": " A useful   snippet. ",
            },
            {
                "title": "duplicate",
                "href": "https://example.com/page",
                "body": "ignored",
            },
            {"title": "bad", "href": "javascript:alert(1)", "body": "ignored"},
        ]
    )
    gateway = DuckDuckGoSearchGateway(
        timeout_seconds=4.0,
        client_factory=lambda **kwargs: client,
    )

    results = await gateway.search(
        WebSearchRequest(
            query="current topic",
            region="cn-zh",
            safesearch=WebSearchSafeSearch.MODERATE,
            max_results=5,
            time_range=WebSearchTimeRange.WEEK,
        )
    )

    assert client.thread_id is not None
    assert client.thread_id != main_thread_id
    assert client.arguments == {
        "query": "current topic",
        "region": "cn-zh",
        "safesearch": "moderate",
        "timelimit": "w",
        "max_results": 5,
        "backend": "auto",
    }
    assert results == (
        WebSearchResult(
            title="Example result",
            url="https://example.com/page",
            snippet="A useful snippet.",
        ),
    )
    await gateway.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_error", "expected_error"),
    [
        (TimeoutException("slow"), WebSearchTimeoutError),
        (RatelimitException("busy"), WebSearchRateLimitError),
    ],
)
async def test_duckduckgo_gateway_maps_expected_provider_errors(
    provider_error: Exception,
    expected_error: type[Exception],
) -> None:
    client = RecordingDdgs(error=provider_error)
    gateway = DuckDuckGoSearchGateway(
        timeout_seconds=1.0,
        client_factory=lambda **kwargs: client,
    )
    request = WebSearchRequest(
        query="topic",
        region="cn-zh",
        safesearch=WebSearchSafeSearch.MODERATE,
        max_results=5,
    )

    with pytest.raises(expected_error):
        await gateway.search(request)
    await gateway.aclose()


@pytest.mark.asyncio
async def test_search_service_normalizes_query_and_applies_settings() -> None:
    gateway = StaticGateway((WebSearchResult("Title", "https://example.com", "Snippet"),))
    service = WebSearchService(
        web_settings(
            CYWL_WEB_SEARCH_REGION="us-en",
            CYWL_WEB_SEARCH_SAFESEARCH="off",
            CYWL_WEB_SEARCH_MAX_RESULTS="3",
        ),
        gateway,
    )

    results = await service.search("  current   topic ", time_range=WebSearchTimeRange.MONTH)

    assert results[0].title == "Title"
    assert gateway.request == WebSearchRequest(
        query="current topic",
        region="us-en",
        safesearch=WebSearchSafeSearch.OFF,
        max_results=3,
        time_range=WebSearchTimeRange.MONTH,
    )

    with pytest.raises(WebSearchQueryError):
        await service.search(" ")


@pytest.mark.asyncio
async def test_search_web_tool_returns_bounded_model_shape_and_stable_error() -> None:
    service = WebSearchService(
        web_settings(),
        StaticGateway((WebSearchResult("Title", "https://example.com", "Snippet"),)),
    )
    tool = SearchWebTool(service, timeout_seconds=5, max_output_characters=4000)

    output = await tool.execute(
        tool_context(),
        SearchWebInput(query="  latest   topic ", time_range="d"),
    )

    assert tool.descriptor.display_name == "搜索公开网页"
    assert output.model_dump() == {
        "query": "latest topic",
        "results": (
            {
                "title": "Title",
                "url": "https://example.com",
                "snippet": "Snippet",
            },
        ),
    }

    failing = SearchWebTool(
        WebSearchService(web_settings(), StaticGateway()),
        timeout_seconds=5,
        max_output_characters=4000,
    )
    failing._search = WebSearchService(
        web_settings(CYWL_WEB_SEARCH_MAX_QUERY_CHARACTERS="1"),
        StaticGateway(),
    )
    with pytest.raises(ToolExecutionError, match="invalid_web_search_query"):
        await failing.execute(tool_context(), SearchWebInput(query="too long"))


@pytest.mark.asyncio
async def test_real_duckduckgo_search_when_explicitly_enabled() -> None:
    if os.getenv("CYWL_RUN_DDGS_LIVE_TESTS") != "1":
        pytest.skip("set CYWL_RUN_DDGS_LIVE_TESTS=1 to run live DuckDuckGo search")

    gateway = DuckDuckGoSearchGateway(timeout_seconds=8)
    results = await asyncio.wait_for(
        gateway.search(
            WebSearchRequest(
                query="Python programming language official website",
                region="wt-wt",
                safesearch=WebSearchSafeSearch.MODERATE,
                max_results=3,
            )
        ),
        timeout=10,
    )

    assert results
    assert all(result.url.startswith(("http://", "https://")) for result in results)
    await gateway.aclose()
