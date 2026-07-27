"""Public URL policy and conversation-scoped browser session lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TypeVar
from urllib.parse import urlsplit

from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.settings import WebToolsSettings

from .errors import BrowserError, BrowserUnavailableError, WebPageUrlError
from .models import (
    BrowserActionResult,
    BrowserDocument,
    BrowserPageView,
    BrowserWaitRequest,
)
from .ports import BrowserGateway

_Result = TypeVar("_Result")


class PublicWebUrlPolicy:
    """Accept ordinary public HTTP(S) URLs without performing DNS resolution."""

    def __init__(self, *, max_characters: int = 2048) -> None:
        self._max_characters = max_characters

    def validate(self, url: str) -> str:
        """Return a stripped URL or raise a stable project error."""
        normalized = url.strip()
        if (
            not normalized
            or len(normalized) > self._max_characters
            or any(character.isspace() or ord(character) < 32 for character in normalized)
        ):
            raise WebPageUrlError
        try:
            parsed = urlsplit(normalized)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise WebPageUrlError from exc
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or hostname is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise WebPageUrlError
        del port
        normalized_hostname = hostname.rstrip(".").casefold()
        if normalized_hostname == "localhost" or normalized_hostname.endswith(".localhost"):
            raise WebPageUrlError
        try:
            address = ipaddress.ip_address(normalized_hostname)
        except ValueError:
            return normalized
        if not address.is_global:
            raise WebPageUrlError
        return normalized


@dataclass(slots=True)
class _BrowserSession:
    name: str
    last_used: float
    current_url: str = ""
    started: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class BrowserSessionManager:
    """Serialize browser state per conversation and bound global browser work."""

    def __init__(
        self,
        settings: WebToolsSettings,
        gateway: BrowserGateway,
        *,
        url_policy: PublicWebUrlPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._url_policy = url_policy or PublicWebUrlPolicy()
        self._clock = clock
        self._sessions: dict[ConversationKey, _BrowserSession] = {}
        self._sessions_lock = asyncio.Lock()
        self._operation_slots = asyncio.Semaphore(settings.browser_max_concurrency)
        self._closed = False

    async def start(self) -> None:
        """Start and validate the shared MCP provider."""
        await self._gateway.start()

    async def read(self, key: ConversationKey, url: str) -> BrowserDocument:
        """Read a validated public URL in this conversation's isolated session."""
        validated_url = self._url_policy.validate(url)
        return await self._run(
            key,
            lambda session: self._gateway.read(session, validated_url),
        )

    async def open(self, key: ConversationKey, url: str) -> BrowserPageView:
        """Navigate the conversation's session to a validated public URL."""
        validated_url = self._url_policy.validate(url)
        return await self._run(
            key,
            lambda session: self._gateway.open(session, validated_url),
        )

    async def snapshot(
        self,
        key: ConversationKey,
        *,
        interactive: bool = True,
        compact: bool = True,
    ) -> BrowserPageView:
        """Inspect the current page without accepting a model-controlled session."""
        return await self._run(
            key,
            lambda session: self._gateway.snapshot(
                session,
                interactive=interactive,
                compact=compact,
            ),
        )

    async def wait(
        self,
        key: ConversationKey,
        request: BrowserWaitRequest,
    ) -> BrowserPageView:
        """Wait in the current page and always return a fresh snapshot."""
        return await self._run(
            key,
            lambda session: self._gateway.wait(session, request),
        )

    async def click(self, key: ConversationKey, ref: str) -> BrowserPageView:
        """Click a fresh snapshot ref without retrying the write."""
        return await self._run(
            key,
            lambda session: self._gateway.click(session, ref),
            retry_unavailable=False,
        )

    async def fill(
        self,
        key: ConversationKey,
        ref: str,
        text: str,
    ) -> BrowserActionResult:
        """Fill a fresh snapshot ref without retrying the write."""
        return await self._run(
            key,
            lambda session: self._gateway.fill(session, ref, text),
            retry_unavailable=False,
        )

    async def press(self, key: ConversationKey, key_name: str) -> BrowserPageView:
        """Press an allowed key without retrying the write."""
        return await self._run(
            key,
            lambda session: self._gateway.press(session, key_name),
            retry_unavailable=False,
        )

    async def close(self, key: ConversationKey) -> bool:
        """Close and forget one conversation session."""
        async with self._sessions_lock:
            entry = self._sessions.pop(key, None)
        if entry is None:
            return False
        async with entry.lock:
            async with self._operation_slots:
                await self._gateway.close_session(entry.name)
        return True

    async def _run(
        self,
        key: ConversationKey,
        operation: Callable[[str], Awaitable[_Result]],
        *,
        retry_unavailable: bool = True,
    ) -> _Result:
        if self._closed:
            raise BrowserUnavailableError
        await self._prune_idle()
        entry = await self._session_for(key)
        async with entry.lock:
            async with self._operation_slots:
                try:
                    result = await operation(entry.name)
                except BrowserUnavailableError:
                    if not retry_unavailable:
                        raise
                    await self._gateway.restart()
                    if entry.started and entry.current_url:
                        await self._gateway.open(entry.name, entry.current_url)
                    result = await operation(entry.name)
                if isinstance(
                    result,
                    (BrowserActionResult, BrowserDocument, BrowserPageView),
                ):
                    entry.current_url = result.url
                    entry.started = True
                entry.last_used = self._clock()
                return result

    async def _session_for(self, key: ConversationKey) -> _BrowserSession:
        async with self._sessions_lock:
            entry = self._sessions.get(key)
            if entry is None:
                entry = _BrowserSession(
                    name=self.session_name(key),
                    last_used=self._clock(),
                )
                self._sessions[key] = entry
            return entry

    async def _prune_idle(self) -> None:
        cutoff = self._clock() - self._settings.browser_session_idle_seconds
        expired: list[_BrowserSession] = []
        async with self._sessions_lock:
            for key, entry in tuple(self._sessions.items()):
                if entry.last_used <= cutoff and not entry.lock.locked():
                    expired.append(self._sessions.pop(key))
        for entry in expired:
            async with self._operation_slots:
                with suppress(BrowserError):
                    await self._gateway.close_session(entry.name)

    @staticmethod
    def session_name(key: ConversationKey) -> str:
        """Hash raw OOPZ identifiers into a short Unix-socket-safe name."""
        material = "\0".join((key.scope, key.area_id, key.channel_id, key.person_id))
        digest = hashlib.blake2s(material.encode(), digest_size=10).hexdigest()
        return f"cywl-{digest}"

    async def aclose(self) -> None:
        """Close known sessions before closing the shared MCP transport."""
        if self._closed:
            return
        self._closed = True
        async with self._sessions_lock:
            entries = tuple(self._sessions.values())
            self._sessions.clear()
        for entry in entries:
            async with entry.lock:
                async with self._operation_slots:
                    with suppress(BrowserError):
                        await self._gateway.close_session(entry.name)
        await self._gateway.aclose()
