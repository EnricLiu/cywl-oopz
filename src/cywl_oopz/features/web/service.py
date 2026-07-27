"""Application policy for bounded web search."""

from __future__ import annotations

import logging

from cywl_oopz.settings import WebToolsSettings

from .errors import WebSearchQueryError
from .models import WebSearchRequest, WebSearchResult, WebSearchTimeRange
from .ports import WebSearchGateway

logger = logging.getLogger(__name__)


class WebSearchService:
    """Validate model queries and apply deployment-local provider defaults."""

    def __init__(self, settings: WebToolsSettings, gateway: WebSearchGateway) -> None:
        self._settings = settings
        self._gateway = gateway

    async def search(
        self,
        query: str,
        *,
        time_range: WebSearchTimeRange | None = None,
    ) -> tuple[WebSearchResult, ...]:
        """Search with whitespace normalization and strict input bounds."""
        normalized = " ".join(query.split())
        if not normalized or len(normalized) > self._settings.search_max_query_characters:
            logger.warning("Rejected invalid web search query: characters=%s", len(normalized))
            raise WebSearchQueryError
        logger.info(
            "Web search started: query_characters=%s time_range=%s max_results=%s",
            len(normalized),
            time_range.value if time_range is not None else "none",
            self._settings.search_max_results,
        )
        results = await self._gateway.search(
            WebSearchRequest(
                query=normalized,
                region=self._settings.search_region,
                safesearch=self._settings.search_safesearch,
                max_results=self._settings.search_max_results,
                time_range=time_range,
            )
        )
        logger.info("Web search completed: result_count=%s", len(results))
        return results

    async def aclose(self) -> None:
        """Release provider resources."""
        logger.debug("Closing web search service")
        await self._gateway.aclose()
