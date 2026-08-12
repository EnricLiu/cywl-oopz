"""Bilibili video metadata and playback through the isolated yt-dlp worker."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

from cywl_oopz.integrations.media.ytdlp_models import (
    YtDlpMode,
    YtDlpOperation,
    YtDlpWorkerItem,
)
from cywl_oopz.integrations.media.ytdlp_runner import YtDlpProcessRunner
from cywl_oopz.settings import MusicSettings, YtDlpMusicSettings

from .errors import MusicReferenceError, MusicUnsupportedContentError
from .models import (
    MusicPageLocator,
    MusicSourceKind,
    MusicTrack,
    MusicTrackReference,
    PlayableTrack,
    ResolvedMediaInput,
)
from .references import MusicInputParser
from .ytdlp_provider import YtDlpMusicProviderBase

_BVID = re.compile(r"BV[A-Za-z0-9]{6,32}\Z", re.IGNORECASE)
_WORKER_ID = re.compile(
    r"(?P<bvid>BV[A-Za-z0-9]{6,32})(?:_p(?P<page>[1-9][0-9]{0,5}))?\Z",
    re.IGNORECASE,
)
_SOURCE_ID = re.compile(
    r"(?P<bvid>BV[A-Za-z0-9]{6,32}):p=(?P<page>[1-9][0-9]{0,5})\Z",
    re.IGNORECASE,
)
_BILIBILI_PAGE_HOSTS = frozenset({"bilibili.com", "www.bilibili.com"})


class BilibiliMusicProvider(YtDlpMusicProviderBase):
    """Expose only ordinary, bounded, non-live Bilibili video parts."""

    source = MusicSourceKind.BILIBILI
    profile = "bilibili_public"
    expected_extractors = ("BiliBili",)

    def __init__(
        self,
        settings: MusicSettings,
        ytdlp_settings: YtDlpMusicSettings,
        runner: YtDlpProcessRunner,
    ) -> None:
        super().__init__(
            settings,
            runner,
            cookie_file=ytdlp_settings.bilibili_cookie_file,
            require_javascript=False,
        )
        self._input_parser = MusicInputParser()

    async def search(self, query: str, *, limit: int) -> tuple[MusicTrack, ...]:
        items = await self._extract(
            YtDlpOperation.SEARCH,
            YtDlpMode.FLAT_SEARCH,
            f"bilisearch{limit}:{query}",
            limit=limit,
        )
        tracks: list[MusicTrack] = []
        for item in items:
            tracks.append(self._track(item, require_duration=False))
        return tuple(tracks)

    async def lookup(self, reference: MusicTrackReference) -> MusicTrack:
        source_id = self._normalize_reference(reference)
        item = await self._one(
            YtDlpOperation.LOOKUP,
            YtDlpMode.FULL_METADATA,
            self._canonical_url(source_id),
        )
        track = self._track(item, require_duration=True)
        if track.source_id != source_id:
            raise MusicReferenceError("Bilibili video part did not match its reference")
        return track

    async def inspect(self, locator: MusicPageLocator) -> MusicTrack:
        if locator.source is not self.source:
            raise MusicReferenceError("Bilibili page locator has the wrong source")
        parsed = self._input_parser.parse(locator.url)
        if not isinstance(parsed, MusicPageLocator) or parsed.source is not self.source:
            raise MusicReferenceError("Bilibili page does not require remote inspection")
        item = await self._one(
            YtDlpOperation.INSPECT,
            YtDlpMode.FULL_METADATA,
            locator.url,
        )
        self._validate_final_page(item)
        return self._track(item, require_duration=True)

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        source_id = self._normalize_reference(track.reference)
        item = await self._one(
            YtDlpOperation.RESOLVE,
            YtDlpMode.PLAYABLE_MEDIA,
            self._canonical_url(source_id),
        )
        trusted = self._track(item, require_duration=True)
        if trusted.source_id != source_id:
            raise MusicReferenceError("Bilibili video part did not match its reference")
        if item.media is None:
            raise MusicUnsupportedContentError("Bilibili returned no playable media")
        media = item.media
        return PlayableTrack(
            track,
            ResolvedMediaInput(
                media.url,
                media.http_headers,
                protocol=media.protocol,
                format_id=media.format_id,
                container=media.container,
                audio_codec=media.audio_codec,
            ),
        )

    async def _one(
        self,
        operation: YtDlpOperation,
        mode: YtDlpMode,
        target: str,
    ) -> YtDlpWorkerItem:
        items = await self._extract(operation, mode, target)
        if len(items) != 1:
            raise MusicUnsupportedContentError("Bilibili returned a media collection")
        return items[0]

    def _track(self, item: YtDlpWorkerItem, *, require_duration: bool) -> MusicTrack:
        source_id = self._source_id(item)
        duration_ms = self._validate_content(item, require_duration=require_duration)
        artists = (item.uploader,) if item.uploader else item.artists
        return MusicTrack(
            self.source,
            source_id,
            item.title,
            artists,
            duration_ms,
        )

    @classmethod
    def _source_id(cls, item: YtDlpWorkerItem) -> str:
        match = _WORKER_ID.fullmatch(item.id)
        page_from_url: int | None = None
        bvid_from_url: str | None = None
        if item.webpage_url:
            parsed = urlsplit(item.webpage_url)
            segments = tuple(segment for segment in parsed.path.split("/") if segment)
            if (
                (parsed.hostname or "").casefold().rstrip(".") in _BILIBILI_PAGE_HOSTS
                and len(segments) >= 2
                and segments[0].casefold() == "video"
                and _BVID.fullmatch(segments[1])
            ):
                bvid_from_url = cls._normalize_bvid(segments[1])
                raw_page = parse_qs(parsed.query).get("p", ("1",))[0]
                if raw_page.isdigit() and int(raw_page) > 0:
                    page_from_url = int(raw_page)
        if match is None and bvid_from_url is None:
            raise MusicReferenceError("Bilibili result has no stable BVID")
        bvid = cls._normalize_bvid(match.group("bvid")) if match else bvid_from_url
        assert bvid is not None
        if bvid_from_url is not None and bvid_from_url != bvid:
            raise MusicReferenceError("Bilibili result identifiers disagree")
        worker_page = int(match.group("page")) if match and match.group("page") else None
        if worker_page is not None and page_from_url is not None and worker_page != page_from_url:
            raise MusicReferenceError("Bilibili result video parts disagree")
        return f"{bvid}:p={worker_page or page_from_url or 1}"

    @classmethod
    def _normalize_reference(cls, reference: MusicTrackReference) -> str:
        if reference.source is not cls.source:
            raise MusicReferenceError("Bilibili reference has the wrong source")
        match = _SOURCE_ID.fullmatch(reference.source_id)
        if match is None:
            raise MusicReferenceError("Bilibili reference must contain a BVID and video part")
        return f"{cls._normalize_bvid(match.group('bvid'))}:p={int(match.group('page'))}"

    @classmethod
    def _canonical_url(cls, source_id: str) -> str:
        match = _SOURCE_ID.fullmatch(source_id)
        assert match is not None
        return (
            f"https://www.bilibili.com/video/{cls._normalize_bvid(match.group('bvid'))}"
            f"?p={int(match.group('page'))}"
        )

    @staticmethod
    def _normalize_bvid(value: str) -> str:
        return f"BV{value[2:]}"

    @staticmethod
    def _validate_final_page(item: YtDlpWorkerItem) -> None:
        if item.webpage_url is None:
            raise MusicReferenceError("Bilibili inspection returned no final page")
        parsed = urlsplit(item.webpage_url)
        host = (parsed.hostname or "").casefold().rstrip(".")
        segments = tuple(segment for segment in parsed.path.split("/") if segment)
        if (
            host not in _BILIBILI_PAGE_HOSTS
            or len(segments) < 2
            or segments[0].casefold() != "video"
            or not (_BVID.fullmatch(segments[1]) or segments[1].casefold().startswith("av"))
        ):
            raise MusicReferenceError("Bilibili locator did not resolve to an allowed video page")
