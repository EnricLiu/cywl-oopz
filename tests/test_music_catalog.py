from __future__ import annotations

import json

import httpx
import pytest

from cywl_oopz.features.music.errors import (
    MusicCatalogError,
    MusicNotFoundError,
    NeteasePlaylistReferenceError,
)
from cywl_oopz.features.music.models import (
    MusicSourceKind,
    MusicTrack,
    MusicTrackReference,
    NeteasePlaylistSnapshot,
)
from cywl_oopz.features.music.netease import (
    NeteaseMusicCatalog,
    NeteasePlaylistReference,
)
from cywl_oopz.settings import MusicSettings


def settings() -> MusicSettings:
    return MusicSettings.from_mapping(
        {
            "CYWL_MUSIC_ENABLED": "true",
            "CYWL_MUSIC_CATALOG_BASE_URL": "https://music.example",
            "CYWL_MUSIC_SEARCH_LIMIT": "5",
        }
    )


@pytest.mark.asyncio
async def test_netease_catalog_searches_and_resolves_stream_at_playback_time() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "songs": [
                            {
                                "id": 42,
                                "name": "Blue Train",
                                "artists": [{"name": "John Coltrane"}],
                                "duration": 640000,
                            }
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            json={"data": [{"id": 42, "url": "https://cdn.example/blue.mp3"}]},
        )

    client = httpx.AsyncClient(
        base_url="https://music.example",
        transport=httpx.MockTransport(respond),
    )
    catalog = NeteaseMusicCatalog(settings(), client)

    tracks = await catalog.search("Blue Train", limit=3)
    playable = await catalog.resolve(tracks[0])

    assert tracks == (
        MusicTrack(
            source="netease",
            source_id="42",
            title="Blue Train",
            artists=("John Coltrane",),
            duration_ms=640000,
        ),
    )
    assert playable.stream_url == "https://cdn.example/blue.mp3"
    assert requests[0].url.params["keywords"] == "Blue Train"
    assert requests[1].url.params["br"] == "320000"
    await client.aclose()


@pytest.mark.parametrize(
    "reference",
    (
        "24381616",
        "https://music.163.com/playlist?id=24381616",
        "https://music.163.com/#/playlist?id=24381616",
        "https://y.music.163.com/m/playlist?id=24381616",
    ),
)
def test_netease_playlist_reference_accepts_ids_and_canonical_urls(reference: str) -> None:
    assert NeteasePlaylistReference.parse(reference).playlist_id == "24381616"


@pytest.mark.parametrize(
    "reference",
    (
        "",
        "not-a-playlist",
        "https://example.com/playlist?id=24381616",
        "ftp://music.163.com/playlist?id=24381616",
        "https://music.163.com/playlist",
        "1" * 21,
    ),
)
def test_netease_playlist_reference_rejects_noncanonical_values(reference: str) -> None:
    with pytest.raises(NeteasePlaylistReferenceError):
        NeteasePlaylistReference.parse(reference)


@pytest.mark.asyncio
async def test_netease_catalog_reads_playlist_metadata_and_complete_track_endpoint() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/playlist/detail":
            return httpx.Response(
                200,
                json={
                    "playlist": {
                        "name": "Miku Favorites",
                        "trackCount": 2,
                        "tracks": [{"id": 1, "name": "incomplete"}],
                        "trackIds": [{"id": 39}, {"id": 831}],
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "songs": [
                    {
                        "id": 39,
                        "name": "39",
                        "ar": [{"name": "初音未来"}],
                        "dt": 222000,
                    },
                    {
                        "id": 831,
                        "name": "Tell Your World",
                        "ar": [{"name": "初音未来"}],
                        "dt": 245000,
                    },
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="https://music.example",
        transport=httpx.MockTransport(respond),
    )
    catalog = NeteaseMusicCatalog(settings(), client)

    playlist = await catalog.playlist(
        "https://music.163.com/#/playlist?id=24381616",
        limit=50,
    )

    assert playlist == NeteasePlaylistSnapshot(
        "24381616",
        "Miku Favorites",
        2,
        (
            MusicTrack("netease", "39", "39", ("初音未来",), 222000),
            MusicTrack(
                "netease",
                "831",
                "Tell Your World",
                ("初音未来",),
                245000,
            ),
        ),
    )
    assert requests[0].url.path == "/playlist/detail"
    assert requests[1].url.path == "/playlist/track/all"
    assert requests[1].url.params["limit"] == "50"
    assert requests[1].url.params["offset"] == "0"
    await client.aclose()


@pytest.mark.asyncio
async def test_netease_catalog_rejects_empty_or_invalid_results() -> None:
    responses = iter(
        (
            httpx.Response(200, json={"result": {"songs": []}}),
            httpx.Response(200, content=json.dumps(["invalid"])),
        )
    )
    client = httpx.AsyncClient(
        base_url="https://music.example",
        transport=httpx.MockTransport(lambda request: next(responses)),
    )
    catalog = NeteaseMusicCatalog(settings(), client)

    with pytest.raises(MusicNotFoundError):
        await catalog.search("missing", limit=1)
    with pytest.raises(MusicCatalogError):
        await catalog.search("invalid", limit=1)

    await client.aclose()


@pytest.mark.asyncio
async def test_netease_catalog_looks_up_one_stable_song_reference() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "songs": [
                    {
                        "id": 831,
                        "name": "Tell Your World",
                        "ar": [{"name": "初音未来"}],
                        "dt": 245000,
                    }
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="https://music.example",
        transport=httpx.MockTransport(respond),
    )
    catalog = NeteaseMusicCatalog(settings(), client)

    track = await catalog.lookup(MusicTrackReference(MusicSourceKind.NETEASE, "831"))

    assert track == MusicTrack(
        MusicSourceKind.NETEASE,
        "831",
        "Tell Your World",
        ("初音未来",),
        245000,
    )
    assert requests[0].url.path == "/song/detail"
    assert requests[0].url.params["ids"] == "831"
    await client.aclose()
