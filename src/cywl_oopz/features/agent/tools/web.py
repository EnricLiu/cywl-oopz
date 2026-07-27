"""Agent tool adapters for project-owned web capabilities."""

from __future__ import annotations

from typing import Literal, Never

from pydantic import BaseModel, ConfigDict, Field

from cywl_oopz.features.web.errors import (
    WebSearchError,
    WebSearchQueryError,
    WebSearchRateLimitError,
    WebSearchTimeoutError,
    WebSearchUnavailableError,
)
from cywl_oopz.features.web.models import WebSearchResult, WebSearchTimeRange
from cywl_oopz.features.web.service import WebSearchService

from .models import ToolDescriptor, ToolEffect, ToolExecutionContext, ToolExecutionError


class SearchWebInput(BaseModel):
    """A bounded public-web query with an optional recency filter."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=300, description="要检索的具体问题或关键词")
    time_range: Literal["d", "w", "m", "y"] | None = Field(
        default=None,
        description="可选时间范围：一天、一周、一月或一年",
    )


class WebSearchResultOutput(BaseModel):
    """One normalized search result."""

    title: str
    url: str
    snippet: str

    @classmethod
    def from_result(cls, result: WebSearchResult) -> WebSearchResultOutput:
        return cls(title=result.title, url=result.url, snippet=result.snippet)


class SearchWebOutput(BaseModel):
    """Ordered search results with the effective normalized query."""

    query: str
    results: tuple[WebSearchResultOutput, ...]


class SearchWebTool:
    """Search the public web without granting raw network or browser access."""

    _ERROR_CODES = {
        WebSearchQueryError: "invalid_web_search_query",
        WebSearchTimeoutError: "web_search_timeout",
        WebSearchRateLimitError: "web_search_rate_limited",
        WebSearchUnavailableError: "web_search_unavailable",
    }

    def __init__(
        self,
        search: WebSearchService,
        *,
        timeout_seconds: float,
        max_output_characters: int,
    ) -> None:
        self._search = search
        self._descriptor = ToolDescriptor(
            name="search_web",
            display_name="搜索网页",
            description=(
                "使用 DuckDuckGo 搜索公开网页，适合获取当前信息或为事实回答查找来源；"
                "结果包含标题、链接和摘要。"
            ),
            input_model=SearchWebInput,
            output_model=SearchWebOutput,
            effect=ToolEffect.READ,
            timeout_seconds=timeout_seconds,
            max_output_characters=max_output_characters,
            concurrency_safe=True,
            idempotent=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        del context
        values = SearchWebInput.model_validate(arguments)
        try:
            results = await self._search.search(
                values.query,
                time_range=(
                    WebSearchTimeRange(values.time_range) if values.time_range is not None else None
                ),
            )
        except WebSearchError as exc:
            self._raise_tool_error(exc)
        return SearchWebOutput(
            query=" ".join(values.query.split()),
            results=tuple(WebSearchResultOutput.from_result(result) for result in results),
        )

    @classmethod
    def _raise_tool_error(cls, error: WebSearchError) -> Never:
        for error_type, code in cls._ERROR_CODES.items():
            if isinstance(error, error_type):
                raise ToolExecutionError(code) from error
        raise ToolExecutionError("web_search_failed") from error
