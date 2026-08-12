"""Pure parsing of user music input into provider-owned references."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from .errors import MusicReferenceError
from .models import MusicPageLocator, MusicSourceKind, MusicTrackReference

_YOUTUBE_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_BILIBILI_BVID = re.compile(r"BV[A-Za-z0-9]{6,32}\Z", re.IGNORECASE)
_BILIBILI_AVID = re.compile(r"av[0-9]{1,20}\Z", re.IGNORECASE)
_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)
_BILIBILI_HOSTS = frozenset({"bilibili.com", "www.bilibili.com", "b23.tv"})


class MusicInputParser:
    """Classify plain queries, stable URLs, and locators without network I/O."""

    def __init__(self, *, max_url_characters: int = 2_048) -> None:
        if max_url_characters <= 0:
            raise ValueError("Music URL limit must be positive")
        self._max_url_characters = max_url_characters

    def parse(self, value: str) -> MusicTrackReference | MusicPageLocator | None:
        normalized = value.strip()
        if not normalized:
            return None
        try:
            parsed = urlsplit(normalized)
        except ValueError as exc:
            raise MusicReferenceError("Music URL is invalid") from exc
        if parsed.scheme.casefold() not in {"http", "https"}:
            if "://" in normalized:
                raise MusicReferenceError("Music URL must use HTTP or HTTPS")
            return None
        if len(normalized) > self._max_url_characters:
            raise MusicReferenceError("Music URL is too long")
        if parsed.username is not None or parsed.password is not None:
            raise MusicReferenceError("Music URL must not contain credentials")
        host = (parsed.hostname or "").casefold().rstrip(".")
        if host in _YOUTUBE_HOSTS:
            return self._youtube(parsed.path, parsed.query)
        if host in _BILIBILI_HOSTS:
            return self._bilibili(normalized, host, parsed.path, parsed.query)
        if host == "music.163.com" or host.endswith(".music.163.com"):
            return self._netease(parsed.path, parsed.query, parsed.fragment)
        raise MusicReferenceError("Music URL source is not supported")

    @staticmethod
    def _youtube(path: str, query: str) -> MusicTrackReference:
        segments = tuple(segment for segment in path.split("/") if segment)
        if segments and segments[0].casefold() in {"shorts", "live"}:
            video_id = segments[1] if len(segments) > 1 else ""
        elif segments and segments[0].casefold() == "watch":
            video_id = parse_qs(query).get("v", ("",))[0]
        elif len(segments) == 1:
            video_id = segments[0]
        elif segments and segments[0].casefold() == "playlist":
            raise MusicReferenceError("YouTube playlist import is not supported yet")
        else:
            video_id = ""
        if not _YOUTUBE_ID.fullmatch(video_id):
            raise MusicReferenceError("YouTube URL has no valid video ID")
        return MusicTrackReference(MusicSourceKind.YOUTUBE, video_id)

    @staticmethod
    def _bilibili(
        original_url: str,
        host: str,
        path: str,
        query: str,
    ) -> MusicTrackReference | MusicPageLocator:
        if host == "b23.tv":
            if not path.strip("/"):
                raise MusicReferenceError("Bilibili short URL has no target")
            return MusicPageLocator(MusicSourceKind.BILIBILI, original_url)
        segments = tuple(segment for segment in path.split("/") if segment)
        if len(segments) < 2 or segments[0].casefold() != "video":
            raise MusicReferenceError("Bilibili URL is not a supported video page")
        identifier = segments[1]
        if _BILIBILI_AVID.fullmatch(identifier):
            return MusicPageLocator(MusicSourceKind.BILIBILI, original_url)
        if not _BILIBILI_BVID.fullmatch(identifier):
            raise MusicReferenceError("Bilibili URL has no valid BVID")
        raw_page = parse_qs(query).get("p", ("1",))[0]
        if not raw_page.isdigit() or not 1 <= int(raw_page) <= 100_000:
            raise MusicReferenceError("Bilibili video part must be a positive integer")
        return MusicTrackReference(
            MusicSourceKind.BILIBILI,
            f"{identifier}:p={int(raw_page)}",
        )

    @staticmethod
    def _netease(path: str, query: str, fragment: str) -> MusicTrackReference:
        candidates: list[str] = []
        if path.rstrip("/").endswith("/song"):
            candidates.extend(parse_qs(query).get("id", ()))
        fragment_path, _, fragment_query = fragment.partition("?")
        if fragment_path.rstrip("/").endswith("/song"):
            candidates.extend(parse_qs(fragment_query).get("id", ()))
        for source_id in candidates:
            if source_id.isdigit() and len(source_id) <= 20:
                return MusicTrackReference(MusicSourceKind.NETEASE, source_id)
        raise MusicReferenceError("Netease URL has no valid song ID")
