"""YouTube video metadata and playback through the isolated yt-dlp worker."""

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
from .ytdlp_provider import YtDlpMusicProviderBase

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)


class YouTubeMusicProvider(YtDlpMusicProviderBase):
    """Expose bounded public YouTube videos under one stable video-ID source."""

    source = MusicSourceKind.YOUTUBE
    profile = "youtube_public"
    expected_extractors = ("Youtube",)

    def __init__(
        self,
        settings: MusicSettings,
        ytdlp_settings: YtDlpMusicSettings,
        runner: YtDlpProcessRunner,
    ) -> None:
        super().__init__(
            settings,
            runner,
            cookie_file=ytdlp_settings.youtube_cookie_file,
            require_javascript=True,
            youtube_player_clients=ytdlp_settings.youtube_player_clients,
        )

    async def search(self, query: str, *, limit: int) -> tuple[MusicTrack, ...]:
        items = await self._extract(
            YtDlpOperation.SEARCH,
            YtDlpMode.FLAT_SEARCH,
            f"ytsearch{limit}:{query}",
            limit=limit,
        )
        return tuple(self._track(item, require_duration=False) for item in items)

    async def lookup(self, reference: MusicTrackReference) -> MusicTrack:
        video_id = self._normalize_reference(reference)
        item = await self._one(
            YtDlpOperation.LOOKUP,
            YtDlpMode.FULL_METADATA,
            self._canonical_url(video_id),
        )
        track = self._track(item, require_duration=True)
        if track.source_id != video_id:
            raise MusicReferenceError("YouTube video did not match its reference")
        return track

    async def inspect(self, locator: MusicPageLocator) -> MusicTrack:
        del locator
        raise MusicReferenceError("YouTube pages do not require remote inspection")

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        video_id = self._normalize_reference(track.reference)
        item = await self._one(
            YtDlpOperation.RESOLVE,
            YtDlpMode.PLAYABLE_MEDIA,
            self._canonical_url(video_id),
        )
        trusted = self._track(item, require_duration=True)
        if trusted.source_id != video_id:
            raise MusicReferenceError("YouTube video did not match its reference")
        if item.media is None:
            raise MusicUnsupportedContentError("YouTube returned no playable media")
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
            raise MusicUnsupportedContentError("YouTube returned a media collection")
        return items[0]

    def _track(self, item: YtDlpWorkerItem, *, require_duration: bool) -> MusicTrack:
        video_id = self._video_id(item)
        duration_ms = self._validate_content(item, require_duration=require_duration)
        artists = item.artists
        if not artists:
            name = item.channel or item.uploader
            artists = (name,) if name else ()
        return MusicTrack(
            self.source,
            video_id,
            item.title,
            artists,
            duration_ms,
        )

    @classmethod
    def _video_id(cls, item: YtDlpWorkerItem) -> str:
        if not _VIDEO_ID.fullmatch(item.id):
            raise MusicReferenceError("YouTube result has no stable video ID")
        if item.webpage_url is not None:
            page_id = cls._video_id_from_url(item.webpage_url)
            if page_id != item.id:
                raise MusicReferenceError("YouTube result identifiers disagree")
        return item.id

    @classmethod
    def _normalize_reference(cls, reference: MusicTrackReference) -> str:
        if reference.source is not cls.source:
            raise MusicReferenceError("YouTube reference has the wrong source")
        if not _VIDEO_ID.fullmatch(reference.source_id):
            raise MusicReferenceError("YouTube reference must be an 11-character video ID")
        return reference.source_id

    @staticmethod
    def _canonical_url(video_id: str) -> str:
        return f"https://www.youtube.com/watch?v={video_id}"

    @staticmethod
    def _video_id_from_url(url: str) -> str:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold().rstrip(".")
        if parsed.scheme.casefold() not in {"http", "https"} or host not in _YOUTUBE_HOSTS:
            raise MusicReferenceError("YouTube result has an untrusted final page")
        segments = tuple(segment for segment in parsed.path.split("/") if segment)
        if host == "youtu.be":
            video_id = segments[0] if len(segments) == 1 else ""
        elif segments and segments[0].casefold() in {"shorts", "live"}:
            video_id = segments[1] if len(segments) > 1 else ""
        elif segments and segments[0].casefold() == "watch":
            video_id = parse_qs(parsed.query).get("v", ("",))[0]
        else:
            video_id = ""
        if not _VIDEO_ID.fullmatch(video_id):
            raise MusicReferenceError("YouTube result has no valid final video ID")
        return video_id
