from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cywl_oopz.features.music.bilibili import BilibiliMusicProvider
from cywl_oopz.features.music.errors import (
    MusicAuthenticationRequiredError,
    MusicLiveUnsupportedError,
    MusicNoAudioFormatError,
    MusicNotFoundError,
    MusicReferenceError,
    MusicSourceRateLimitedError,
    MusicTrackTooLongError,
    MusicUnsupportedContentError,
)
from cywl_oopz.features.music.models import (
    MusicPageLocator,
    MusicProviderHealthState,
    MusicSourceKind,
    MusicTrack,
    MusicTrackReference,
)
from cywl_oopz.integrations.media.ytdlp_models import (
    YtDlpMode,
    YtDlpOperation,
    YtDlpWorkerConfiguration,
    YtDlpWorkerError,
    YtDlpWorkerItem,
    YtDlpWorkerMedia,
    YtDlpWorkerRequest,
    YtDlpWorkerResponse,
)
from cywl_oopz.settings import MusicSettings, YtDlpMusicSettings

BVID = "BV17x411w7KC"


@dataclass
class FakeRunner:
    responses: list[YtDlpWorkerResponse]
    requests: list[YtDlpWorkerRequest] = field(default_factory=list)
    probe_response: YtDlpWorkerResponse = field(
        default_factory=lambda: YtDlpWorkerResponse(
            ok=True,
            capabilities={"yt_dlp": "2026.7.4"},
        )
    )

    def configuration(
        self,
        *,
        cookie_file: str = "",
        require_javascript: bool = False,
        youtube_player_clients: tuple[str, ...] = (),
    ) -> YtDlpWorkerConfiguration:
        assert youtube_player_clients == ()
        return YtDlpWorkerConfiguration(
            cookie_file=cookie_file,
            require_javascript=require_javascript,
        )

    async def run(self, request: YtDlpWorkerRequest) -> YtDlpWorkerResponse:
        self.requests.append(request)
        return self.responses.pop(0)

    async def probe(self, *, require_javascript: bool) -> YtDlpWorkerResponse:
        assert require_javascript is False
        return self.probe_response


def settings(*, max_duration: int = 10_800) -> tuple[MusicSettings, YtDlpMusicSettings]:
    music = MusicSettings.from_mapping(
        {
            "CYWL_MUSIC_SOURCES": "bilibili",
            "CYWL_MUSIC_DEFAULT_SOURCE": "bilibili",
            "CYWL_MUSIC_MAX_TRACK_DURATION_SECONDS": str(max_duration),
        }
    )
    return music, YtDlpMusicSettings.from_mapping(
        {"CYWL_MUSIC_BILIBILI_COOKIE_FILE": "/fixture/bilibili.cookies"}
    )


def item(
    *,
    identifier: str = f"{BVID}_p2",
    webpage_url: str | None = f"https://www.bilibili.com/video/{BVID}?p=2",
    duration: float | None = 180.5,
    is_live: bool | None = False,
    live_status: str | None = "not_live",
    media: YtDlpWorkerMedia | None = None,
) -> YtDlpWorkerItem:
    return YtDlpWorkerItem(
        extractor_key="BiliBili",
        id=identifier,
        title="初音未来演唱会 p02 Tell Your World",
        uploader="初音未来公式",
        duration_seconds=duration,
        webpage_url=webpage_url,
        is_live=is_live,
        live_status=live_status,
        media=media,
    )


def provider(*responses: YtDlpWorkerResponse, max_duration: int = 10_800):
    music, ytdlp = settings(max_duration=max_duration)
    runner = FakeRunner(list(responses))
    return BilibiliMusicProvider(music, ytdlp, runner), runner


@pytest.mark.asyncio
async def test_bilibili_search_recovers_bvid_from_flat_page_url() -> None:
    response = YtDlpWorkerResponse(
        ok=True,
        items=(
            item(
                identifier="170001",
                webpage_url=f"https://www.bilibili.com/video/{BVID}",
                duration=90.25,
            ),
        ),
    )
    catalog, runner = provider(response)

    tracks = await catalog.search("初音未来", limit=3)

    assert tracks == (
        MusicTrack(
            MusicSourceKind.BILIBILI,
            f"{BVID}:p=1",
            "初音未来演唱会 p02 Tell Your World",
            ("初音未来公式",),
            90_250,
        ),
    )
    request = runner.requests[0]
    assert request.operation is YtDlpOperation.SEARCH
    assert request.mode is YtDlpMode.FLAT_SEARCH
    assert request.target == "bilisearch3:初音未来"
    assert request.expected_extractors == ("BiliBili",)
    assert request.configuration.cookie_file == "/fixture/bilibili.cookies"
    assert "初音未来" not in repr(request)


@pytest.mark.asyncio
async def test_bilibili_lookup_uses_canonical_part_url_and_trusted_metadata() -> None:
    catalog, runner = provider(YtDlpWorkerResponse(ok=True, items=(item(),)))

    track = await catalog.lookup(MusicTrackReference(MusicSourceKind.BILIBILI, f"{BVID}:p=2"))

    assert track.source_id == f"{BVID}:p=2"
    assert track.duration_ms == 180_500
    assert runner.requests[0].target == (f"https://www.bilibili.com/video/{BVID}?p=2")
    assert runner.requests[0].mode is YtDlpMode.FULL_METADATA


@pytest.mark.asyncio
async def test_bilibili_inspect_normalizes_short_and_old_video_pages() -> None:
    short_catalog, _ = provider(YtDlpWorkerResponse(ok=True, items=(item(),)))
    old_catalog, _ = provider(
        YtDlpWorkerResponse(
            ok=True,
            items=(
                item(
                    webpage_url="https://www.bilibili.com/video/av170001?p=2",
                ),
            ),
        )
    )

    short = await short_catalog.inspect(
        MusicPageLocator(MusicSourceKind.BILIBILI, "https://b23.tv/fixture")
    )
    old = await old_catalog.inspect(
        MusicPageLocator(
            MusicSourceKind.BILIBILI,
            "https://www.bilibili.com/video/av170001?p=2",
        )
    )

    assert short.source_id == f"{BVID}:p=2"
    assert old.source_id == f"{BVID}:p=2"


@pytest.mark.asyncio
async def test_bilibili_resolve_retains_only_allowed_media_transport() -> None:
    media = YtDlpWorkerMedia(
        "https://cdn.example/audio?deadline=sensitive",
        (
            ("Referer", f"https://www.bilibili.com/video/{BVID}"),
            ("Origin", "https://www.bilibili.com"),
            ("User-Agent", "fixture-agent"),
            ("Cookie", "SESSDATA=secret"),
        ),
        protocol="https",
        format_id="30280",
        container="m4a",
        audio_codec="mp4a.40.2",
    )
    catalog, runner = provider(YtDlpWorkerResponse(ok=True, items=(item(media=media),)))
    queued = MusicTrack(
        MusicSourceKind.BILIBILI,
        f"{BVID}:p=2",
        "queued title",
        ("queued uploader",),
    )

    playable = await catalog.resolve(queued)

    assert playable.track is queued
    assert playable.media.format_id == "30280"
    assert playable.media.http_headers == (
        ("User-Agent", "fixture-agent"),
        ("Referer", f"https://www.bilibili.com/video/{BVID}"),
        ("Origin", "https://www.bilibili.com"),
    )
    assert "sensitive" not in repr(playable)
    assert "secret" not in repr(playable)
    assert runner.requests[0].mode is YtDlpMode.PLAYABLE_MEDIA


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "error_type"),
    (
        ("not_found", MusicNotFoundError),
        ("login_required", MusicAuthenticationRequiredError),
        ("rate_limited", MusicSourceRateLimitedError),
        ("no_audio", MusicNoAudioFormatError),
        ("drm", MusicUnsupportedContentError),
        ("ambiguous_collection", MusicUnsupportedContentError),
    ),
)
async def test_bilibili_maps_worker_failures_to_typed_domain_errors(
    code: str,
    error_type: type[Exception],
) -> None:
    catalog, _ = provider(
        YtDlpWorkerResponse(
            ok=False,
            error=YtDlpWorkerError(code, "bounded upstream failure"),
        )
    )

    with pytest.raises(error_type, match="bounded upstream failure"):
        await catalog.lookup(MusicTrackReference(MusicSourceKind.BILIBILI, f"{BVID}:p=2"))


@pytest.mark.asyncio
async def test_bilibili_rejects_live_long_and_mismatched_content() -> None:
    live_catalog, _ = provider(
        YtDlpWorkerResponse(
            ok=True,
            items=(item(is_live=True, live_status="is_live"),),
        )
    )
    long_catalog, _ = provider(
        YtDlpWorkerResponse(ok=True, items=(item(duration=61),)),
        max_duration=60,
    )
    mismatch_catalog, _ = provider(
        YtDlpWorkerResponse(
            ok=True,
            items=(item(identifier=f"{BVID}_p1", webpage_url=None),),
        )
    )
    reference = MusicTrackReference(MusicSourceKind.BILIBILI, f"{BVID}:p=2")

    with pytest.raises(MusicLiveUnsupportedError):
        await live_catalog.lookup(reference)
    with pytest.raises(MusicTrackTooLongError):
        await long_catalog.lookup(reference)
    with pytest.raises(MusicReferenceError, match="did not match"):
        await mismatch_catalog.lookup(reference)


@pytest.mark.asyncio
async def test_bilibili_rejects_untrusted_inspection_and_invalid_reference() -> None:
    untrusted, _ = provider(
        YtDlpWorkerResponse(
            ok=True,
            items=(item(webpage_url="https://example.org/video/fixture"),),
        )
    )

    with pytest.raises(MusicReferenceError, match="allowed video page"):
        await untrusted.inspect(
            MusicPageLocator(MusicSourceKind.BILIBILI, "https://b23.tv/fixture")
        )
    with pytest.raises(MusicReferenceError, match="BVID and video part"):
        await untrusted.lookup(MusicTrackReference(MusicSourceKind.BILIBILI, BVID))


@pytest.mark.asyncio
async def test_bilibili_health_uses_local_runner_capability() -> None:
    catalog, _ = provider()

    health = await catalog.health()

    assert health.source is MusicSourceKind.BILIBILI
    assert health.state is MusicProviderHealthState.READY
