from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cywl_oopz.features.music.catalog import CompositeMusicCatalog, MusicProviderRegistry
from cywl_oopz.features.music.errors import MusicReferenceError, MusicSourceDisabledError
from cywl_oopz.features.music.models import (
    MusicPageLocator,
    MusicProviderHealth,
    MusicProviderHealthState,
    MusicSourceKind,
    MusicTrack,
    MusicTrackReference,
    PlayableTrack,
    ResolvedMediaInput,
)
from cywl_oopz.features.music.references import MusicInputParser


@dataclass
class FakeProvider:
    source: MusicSourceKind
    calls: list[tuple[str, object]] = field(default_factory=list)
    closed: bool = False

    async def search(self, query: str, *, limit: int) -> tuple[MusicTrack, ...]:
        self.calls.append(("search", (query, limit)))
        return (MusicTrack(self.source, f"{self.source.value}-1", query, ("artist",)),)

    async def lookup(self, reference: MusicTrackReference) -> MusicTrack:
        self.calls.append(("lookup", reference))
        return MusicTrack(reference.source, reference.source_id, "lookup", ("artist",))

    async def inspect(self, locator: MusicPageLocator) -> MusicTrack:
        self.calls.append(("inspect", locator.source))
        return MusicTrack(locator.source, "inspected", "inspect", ("artist",))

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        self.calls.append(("resolve", track.reference))
        return PlayableTrack(
            track,
            ResolvedMediaInput(f"https://media.example/{track.source_id}"),
        )

    async def health(self) -> MusicProviderHealth:
        self.calls.append(("health", self.source))
        return MusicProviderHealth(self.source, MusicProviderHealthState.READY)

    async def aclose(self) -> None:
        self.closed = True


def test_music_input_parser_normalizes_supported_stable_urls() -> None:
    parser = MusicInputParser()

    assert parser.parse("Tell Your World") is None
    assert parser.parse("https://youtu.be/dQw4w9WgXcQ") == MusicTrackReference(
        MusicSourceKind.YOUTUBE,
        "dQw4w9WgXcQ",
    )
    assert parser.parse(
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ&list=ignored"
    ) == MusicTrackReference(MusicSourceKind.YOUTUBE, "dQw4w9WgXcQ")
    assert parser.parse("https://www.youtube.com/shorts/dQw4w9WgXcQ") == MusicTrackReference(
        MusicSourceKind.YOUTUBE,
        "dQw4w9WgXcQ",
    )
    assert parser.parse("https://www.bilibili.com/video/BV1bK411W797?p=2") == (
        MusicTrackReference(MusicSourceKind.BILIBILI, "BV1bK411W797:p=2")
    )
    assert parser.parse("https://music.163.com/#/song?id=831") == MusicTrackReference(
        MusicSourceKind.NETEASE,
        "831",
    )


def test_music_input_parser_keeps_unresolved_bilibili_pages_transient() -> None:
    parser = MusicInputParser()

    short = parser.parse("https://b23.tv/abcDEF")
    old = parser.parse("https://www.bilibili.com/video/av170001?p=3")

    assert isinstance(short, MusicPageLocator)
    assert short.source is MusicSourceKind.BILIBILI
    assert isinstance(old, MusicPageLocator)
    assert old.source is MusicSourceKind.BILIBILI
    assert "b23.tv" not in repr(short)


@pytest.mark.parametrize(
    "value",
    (
        "file:///tmp/audio",
        "https://user:secret@youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.example.org/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/playlist?list=PL123",
        "https://b23.tv",
        "https://www.bilibili.com/bangumi/play/ep1",
        "https://www.bilibili.com/video/BV1bK411W797?p=zero",
    ),
)
def test_music_input_parser_rejects_ambiguous_or_unsupported_urls(value: str) -> None:
    with pytest.raises(MusicReferenceError):
        MusicInputParser().parse(value)


@pytest.mark.asyncio
async def test_composite_catalog_routes_every_operation_by_stable_source() -> None:
    netease = FakeProvider(MusicSourceKind.NETEASE)
    youtube = FakeProvider(MusicSourceKind.YOUTUBE)
    registry = MusicProviderRegistry((netease, youtube))
    catalog = CompositeMusicCatalog(registry, MusicSourceKind.NETEASE)

    default_results = await catalog.search("default", limit=2)
    youtube_results = await catalog.search(
        "video",
        source=MusicSourceKind.YOUTUBE,
        limit=3,
    )
    looked_up = await catalog.lookup(MusicTrackReference(MusicSourceKind.YOUTUBE, "video-id"))
    inspected = await catalog.inspect(
        MusicPageLocator(MusicSourceKind.YOUTUBE, "https://youtu.be/dQw4w9WgXcQ")
    )
    playable = await catalog.resolve(youtube_results[0])
    health = await catalog.health()

    assert default_results[0].source is MusicSourceKind.NETEASE
    assert youtube_results[0].source is MusicSourceKind.YOUTUBE
    assert looked_up.source_id == "video-id"
    assert inspected.source_id == "inspected"
    assert playable.media.url.endswith("youtube-1")
    assert [item.source for item in health] == [
        MusicSourceKind.NETEASE,
        MusicSourceKind.YOUTUBE,
    ]
    with pytest.raises(MusicSourceDisabledError):
        await catalog.lookup(MusicTrackReference(MusicSourceKind.BILIBILI, "BV1:p=1"))

    await catalog.aclose()
    assert netease.closed is True
    assert youtube.closed is True


def test_music_provider_registry_rejects_duplicate_and_missing_default_sources() -> None:
    first = FakeProvider(MusicSourceKind.NETEASE)
    with pytest.raises(ValueError, match="Duplicate"):
        MusicProviderRegistry((first, FakeProvider(MusicSourceKind.NETEASE)))
    with pytest.raises(MusicSourceDisabledError):
        CompositeMusicCatalog(MusicProviderRegistry((first,)), MusicSourceKind.YOUTUBE)


def test_resolved_media_input_keeps_transport_details_out_of_repr() -> None:
    media = ResolvedMediaInput(
        "https://media.example/audio?token=sensitive",
        (
            ("referer", "https://secret.example"),
            ("User-Agent", "CYWL/1"),
            ("Cookie", "session=forbidden"),
            ("Authorization", "Bearer forbidden"),
            ("X-Extractor-Internal", "ignored"),
        ),
        protocol="https",
    )

    assert media.http_headers == (
        ("User-Agent", "CYWL/1"),
        ("Referer", "https://secret.example"),
    )
    assert "sensitive" not in repr(media)
    assert "secret.example" not in repr(media)
    assert "forbidden" not in repr(media)


@pytest.mark.parametrize(
    "headers",
    (
        (("Referer", "https://example.com\r\nCookie: injected"),),
        (("User-Agent\nInjected", "value"),),
        (("Origin", "bad\0value"),),
    ),
)
def test_resolved_media_input_rejects_header_injection(
    headers: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(ValueError, match="control characters"):
        ResolvedMediaInput("https://media.example/audio", headers)
