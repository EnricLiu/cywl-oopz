"""Provider-neutral values for internet search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cywl_oopz.settings import WebSearchSafeSearch


class WebSearchTimeRange(StrEnum):
    """DuckDuckGo recency filters kept independent from model-facing labels."""

    DAY = "d"
    WEEK = "w"
    MONTH = "m"
    YEAR = "y"


@dataclass(frozen=True, slots=True)
class WebSearchRequest:
    """One validated provider request."""

    query: str
    region: str
    safesearch: WebSearchSafeSearch
    max_results: int
    time_range: WebSearchTimeRange | None = None


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    """One normalized result safe to expose to an LLM."""

    title: str
    url: str
    snippet: str


@dataclass(frozen=True, slots=True)
class BrowserDocument:
    """Bounded readable content from one public webpage."""

    title: str
    url: str
    content_type: str
    content: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class BrowserPageView:
    """Current page identity and a bounded accessibility snapshot."""

    title: str
    url: str
    snapshot: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class BrowserActionResult:
    """Provider-confirmed result for an action that does not navigate."""

    title: str
    url: str
    applied: bool


class BrowserProgressStage(StrEnum):
    """Provider-neutral milestones emitted by multi-step browser operations."""

    NAVIGATED = "navigated"
    EXTRACTING = "extracting"
    CONTENT_READY = "content_ready"
    SNAPSHOT_READY = "snapshot_ready"
    ACTION_APPLIED = "action_applied"
    IDENTITY_READY = "identity_ready"


@dataclass(frozen=True, slots=True)
class BrowserProgressUpdate:
    """Small real milestone suitable for adaptation into a user-facing view."""

    stage: BrowserProgressStage
    title: str = ""
    url: str = ""
    preview_lines: tuple[str, ...] = ()


class BrowserWaitKind(StrEnum):
    """Supported high-level wait conditions."""

    LOAD = "load"
    TEXT = "text"
    SELECTOR = "selector"
    MILLISECONDS = "milliseconds"


@dataclass(frozen=True, slots=True)
class BrowserWaitRequest:
    """One bounded wait condition independent from MCP tool naming."""

    kind: BrowserWaitKind
    value: str | int
    timeout_seconds: float
