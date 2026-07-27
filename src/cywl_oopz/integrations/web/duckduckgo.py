"""Async project adapter for the synchronous DuckDuckGo search client."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException

from cywl_oopz.features.web.errors import (
    WebSearchRateLimitError,
    WebSearchTimeoutError,
    WebSearchUnavailableError,
)
from cywl_oopz.features.web.models import WebSearchRequest, WebSearchResult

logger = logging.getLogger(__name__)

_MAX_TITLE_CHARACTERS = 160
_MAX_URL_CHARACTERS = 2048
_MAX_SNIPPET_CHARACTERS = 600


class DuckDuckGoSearchGateway:
    """Run DDGS outside the event loop and normalize its untrusted result shape."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_concurrency: int = 3,
        client_factory: Callable[..., Any] = DDGS,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._client_factory = client_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="cywl-web-search",
        )
        self._closed = False

    async def search(self, request: WebSearchRequest) -> tuple[WebSearchResult, ...]:
        """Execute one synchronous DuckDuckGo request in a worker thread."""
        future: Future[tuple[Mapping[str, Any], ...]] | None = None
        try:
            if self._closed:
                logger.warning("Rejected web search because DuckDuckGo gateway is closed")
                raise WebSearchUnavailableError
            logger.debug(
                "Submitting DuckDuckGo worker request: query_characters=%s max_results=%s",
                len(request.query),
                request.max_results,
            )
            future = self._executor.submit(self._search_sync, request)
            raw_results = await self._wait_for_thread(future)
        except asyncio.CancelledError:
            if future is not None:
                future.cancel()
            logger.info("DuckDuckGo worker request cancelled")
            raise
        except TimeoutException as exc:
            logger.warning("DuckDuckGo request timed out: error=%s", type(exc).__name__)
            raise WebSearchTimeoutError from exc
        except RatelimitException as exc:
            logger.warning("DuckDuckGo rate limited request: error=%s", type(exc).__name__)
            raise WebSearchRateLimitError from exc
        except DDGSException as exc:
            logger.warning("DuckDuckGo request unavailable: error=%s", type(exc).__name__)
            raise WebSearchUnavailableError from exc
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("DuckDuckGo worker failed: error=%s", type(exc).__name__)
            raise WebSearchUnavailableError from exc
        results = self._normalize(raw_results, request.max_results)
        logger.debug(
            "DuckDuckGo worker completed: raw_results=%s normalized_results=%s",
            len(raw_results),
            len(results),
        )
        return results

    @staticmethod
    async def _wait_for_thread(
        future: Future[tuple[Mapping[str, Any], ...]],
    ) -> tuple[Mapping[str, Any], ...]:
        """Poll without blocking when a runtime cannot deliver thread wakeups."""
        while not future.done():
            await asyncio.sleep(0.01)
        return future.result()

    async def aclose(self) -> None:
        """Reject new work and release idle search workers without blocking."""
        if self._closed:
            return
        self._closed = True
        logger.info("Closing DuckDuckGo worker pool")
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _search_sync(self, request: WebSearchRequest) -> tuple[Mapping[str, Any], ...]:
        client = self._client_factory(timeout=self._timeout_seconds)
        return tuple(
            client.text(
                request.query,
                region=request.region,
                safesearch=request.safesearch.value,
                timelimit=request.time_range.value if request.time_range else None,
                max_results=request.max_results,
                backend="auto",
            )
        )

    @staticmethod
    def _normalize(
        raw_results: Iterable[Mapping[str, Any]],
        limit: int,
    ) -> tuple[WebSearchResult, ...]:
        results: list[WebSearchResult] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            url = str(item.get("href") or item.get("url") or "").strip()
            parsed = urlparse(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or len(url) > _MAX_URL_CHARACTERS
                or url in seen_urls
            ):
                continue
            title = " ".join(str(item.get("title") or "").split())
            snippet = " ".join(str(item.get("body") or item.get("snippet") or "").split())
            if not title:
                title = parsed.netloc
            seen_urls.add(url)
            results.append(
                WebSearchResult(
                    title=title[:_MAX_TITLE_CHARACTERS],
                    url=url,
                    snippet=snippet[:_MAX_SNIPPET_CHARACTERS],
                )
            )
            if len(results) >= limit:
                break
        return tuple(results)
