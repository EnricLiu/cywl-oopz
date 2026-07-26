from __future__ import annotations

import json

import httpx
import pytest

from cywl_oopz.features.music.errors import MusicCatalogError, MusicNotFoundError
from cywl_oopz.features.music.models import MusicTrack
from cywl_oopz.features.music.netease import NeteaseMusicCatalog
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
