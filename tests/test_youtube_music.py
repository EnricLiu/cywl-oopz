from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cywl_oopz.features.music.errors import (
    MusicAuthenticationRequiredError,
    MusicLiveUnsupportedError,
    MusicReferenceError,
    MusicTrackTooLongError,
)
from cywl_oopz.features.music.models import (
    MusicPageLocator,
    MusicProviderHealthState,
    MusicSourceKind,
    MusicTrack,
    MusicTrackReference,
)
from cywl_oopz.features.music.youtube import YouTubeMusicProvider
from cywl_oopz.integrations.media.ytdlp_models import (
    YtDlpMode,
    YtDlpOperation,
    YtDlpWorkerConfiguration,
    YtDlpWorkerItem,
    YtDlpWorkerMedia,
    YtDlpWorkerRequest,
    YtDlpWorkerResponse,
)
from cywl_oopz.settings import MusicSettings, YtDlpMusicSettings

VIDEO_ID = "YE7VzlLtp-4"


@dataclass
class FakeRunner:
    responses: list[YtDlpWorkerResponse]
    requests: list[YtDlpWorkerRequest] = field(default_factory=list)
    probe_response: YtDlpWorkerResponse = field(
        default_factory=lambda: YtDlpWorkerResponse(
            ok=True,
            capabilities={"yt_dlp": "2026.7.4", "node": "24.18.0"},
        )
    )

    def configuration(
        self,
        *,
        cookie_file: str = "",
        require_javascript: bool = False,
        youtube_player_clients: tuple[str, ...] = (),
    ) -> YtDlpWorkerConfiguration:
        return YtDlpWorkerConfiguration(
            js_runtime="node",
            cookie_file=cookie_file,
            require_javascript=require_javascript,
            youtube_player_clients=youtube_player_clients,
        )

    async def run(self, request: YtDlpWorkerRequest) -> YtDlpWorkerResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def probe(self, *, require_javascript: bool) -> YtDlpWorkerResponse:
        assert require_javascript is True
        return self.probe_response


def settings(*, max_duration: int = 10_800) -> tuple[MusicSettings, YtDlpMusicSettings]:
    music = MusicSettings.from_mapping(
        {
            "CYWL_MUSIC_SOURCES": "youtube",
            "CYWL_MUSIC_DEFAULT_SOURCE": "youtube",
            "CYWL_MUSIC_MAX_TRACK_DURATION_SECONDS": str(max_duration),
        }
    )
    return music, YtDlpMusicSettings.from_mapping(
        {
            "CYWL_MUSIC_YTDLP_JS_RUNTIME": "node",
            "CYWL_MUSIC_YOUTUBE_COOKIE_FILE": "/fixture/youtube.cookies",
            "CYWL_MUSIC_YOUTUBE_PLAYER_CLIENTS": "mweb,web_safari",
        }
    )


def item(
    *,
    identifier: str = VIDEO_ID,
    webpage_url: str | None = f"https://www.youtube.com/watch?v={VIDEO_ID}",
    artists: tuple[str, ...] = ("Blender Music",),
    channel: str | None = "Blender",
    duration: float | None = 597,
    is_live: bool | None = False,
    live_status: str | None = "not_live",
    availability: str | None = "public",
    media: YtDlpWorkerMedia | None = None,
) -> YtDlpWorkerItem:
    return YtDlpWorkerItem(
        extractor_key="Youtube",
        id=identifier,
        title="Big Buck Bunny",
        artists=artists,
        channel=channel,
        uploader="Blender Official",
        duration_seconds=duration,
        webpage_url=webpage_url,
        is_live=is_live,
        live_status=live_status,
        availability=availability,
        media=media,
    )


def provider(*responses: YtDlpWorkerResponse, max_duration: int = 10_800):
    music, ytdlp = settings(max_duration=max_duration)
    runner = FakeRunner(list(responses))
    return YouTubeMusicProvider(music, ytdlp, runner), runner


@pytest.mark.asyncio
async def test_youtube_search_maps_stable_video_results_and_runtime_profile() -> None:
    catalog, runner = provider(YtDlpWorkerResponse(ok=True, items=(item(duration=42.25),)))

    tracks = await catalog.search("初音未来", limit=3)

    assert tracks == (
        MusicTrack(
            MusicSourceKind.YOUTUBE,
            VIDEO_ID,
            "Big Buck Bunny",
            ("Blender Music",),
            42_250,
        ),
    )
    request = runner.requests[0]
    assert request.operation is YtDlpOperation.SEARCH
    assert request.mode is YtDlpMode.FLAT_SEARCH
    assert request.target == "ytsearch3:初音未来"
    assert request.expected_extractors == ("Youtube",)
    assert request.configuration.require_javascript is True
    assert request.configuration.cookie_file == "/fixture/youtube.cookies"
    assert request.configuration.youtube_player_clients == ("mweb", "web_safari")
    assert "初音未来" not in repr(request)


@pytest.mark.asyncio
async def test_youtube_search_falls_back_to_channel_when_music_artists_are_absent() -> None:
    catalog, _ = provider(YtDlpWorkerResponse(ok=True, items=(item(artists=()),)))

    tracks = await catalog.search("fixture", limit=1)

    assert tracks[0].artists == ("Blender",)


@pytest.mark.asyncio
async def test_youtube_lookup_rebuilds_canonical_watch_url() -> None:
    catalog, runner = provider(YtDlpWorkerResponse(ok=True, items=(item(),)))

    track = await catalog.lookup(MusicTrackReference(MusicSourceKind.YOUTUBE, VIDEO_ID))

    assert track.source_id == VIDEO_ID
    assert track.duration_ms == 597_000
    assert runner.requests[0].target == (f"https://www.youtube.com/watch?v={VIDEO_ID}")
    assert runner.requests[0].mode is YtDlpMode.FULL_METADATA


@pytest.mark.asyncio
async def test_youtube_resolve_keeps_queued_snapshot_and_safe_media() -> None:
    media = YtDlpWorkerMedia(
        "https://googlevideo.example/audio?expire=sensitive",
        (
            ("User-Agent", "fixture-agent"),
            ("Referer", "https://www.youtube.com/"),
            ("Cookie", "secret"),
        ),
        protocol="https",
        format_id="251",
        container="webm",
        audio_codec="opus",
    )
    catalog, runner = provider(YtDlpWorkerResponse(ok=True, items=(item(media=media),)))
    queued = MusicTrack(
        MusicSourceKind.YOUTUBE,
        VIDEO_ID,
        "queued title",
        ("queued artist",),
    )

    playable = await catalog.resolve(queued)

    assert playable.track is queued
    assert playable.media.format_id == "251"
    assert playable.media.http_headers == (
        ("User-Agent", "fixture-agent"),
        ("Referer", "https://www.youtube.com/"),
    )
    assert "sensitive" not in repr(playable)
    assert "secret" not in repr(playable)
    assert runner.requests[0].mode is YtDlpMode.PLAYABLE_MEDIA


@pytest.mark.asyncio
async def test_youtube_rejects_live_long_auth_and_mismatched_content() -> None:
    live, _ = provider(
        YtDlpWorkerResponse(
            ok=True,
            items=(item(is_live=True, live_status="is_live"),),
        )
    )
    long, _ = provider(
        YtDlpWorkerResponse(ok=True, items=(item(duration=61),)),
        max_duration=60,
    )
    auth, _ = provider(
        YtDlpWorkerResponse(
            ok=True,
            items=(item(availability="needs_auth"),),
        )
    )
    mismatch, _ = provider(
        YtDlpWorkerResponse(
            ok=True,
            items=(item(identifier="BaW_jenozKc", webpage_url=None),),
        )
    )
    reference = MusicTrackReference(MusicSourceKind.YOUTUBE, VIDEO_ID)

    with pytest.raises(MusicLiveUnsupportedError):
        await live.lookup(reference)
    with pytest.raises(MusicTrackTooLongError):
        await long.lookup(reference)
    with pytest.raises(MusicAuthenticationRequiredError):
        await auth.lookup(reference)
    with pytest.raises(MusicReferenceError, match="did not match"):
        await mismatch.lookup(reference)


@pytest.mark.asyncio
async def test_youtube_rejects_invalid_final_page_reference_and_inspection() -> None:
    untrusted, _ = provider(
        YtDlpWorkerResponse(
            ok=True,
            items=(item(webpage_url="https://example.org/watch?v=YE7VzlLtp-4"),),
        )
    )

    with pytest.raises(MusicReferenceError, match="untrusted final page"):
        await untrusted.lookup(MusicTrackReference(MusicSourceKind.YOUTUBE, VIDEO_ID))
    with pytest.raises(MusicReferenceError, match="do not require"):
        await untrusted.inspect(
            MusicPageLocator(
                MusicSourceKind.YOUTUBE,
                f"https://youtu.be/{VIDEO_ID}",
            )
        )
    with pytest.raises(MusicReferenceError, match="11-character"):
        await untrusted.lookup(MusicTrackReference(MusicSourceKind.YOUTUBE, "invalid"))


@pytest.mark.asyncio
async def test_youtube_accepts_music_shorts_live_page_shapes_after_extraction() -> None:
    for webpage_url in (
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
    ):
        catalog, _ = provider(YtDlpWorkerResponse(ok=True, items=(item(webpage_url=webpage_url),)))
        assert (
            await catalog.lookup(MusicTrackReference(MusicSourceKind.YOUTUBE, VIDEO_ID))
        ).source_id == VIDEO_ID


@pytest.mark.asyncio
async def test_youtube_health_requires_javascript_capability() -> None:
    catalog, _ = provider()

    health = await catalog.health()

    assert health.source is MusicSourceKind.YOUTUBE
    assert health.state is MusicProviderHealthState.READY
