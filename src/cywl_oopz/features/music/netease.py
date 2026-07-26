"""Async NeteaseCloudMusicApi-compatible catalog adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from cywl_oopz.settings import MusicSettings

from .errors import MusicCatalogError, MusicNotFoundError
from .models import MusicTrack, PlayableTrack


class NeteaseMusicCatalog:
    """Translate Netease-compatible JSON into project-owned music values."""

    def __init__(
        self,
        settings: MusicSettings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        headers = {"Cookie": settings.catalog_cookie} if settings.catalog_cookie else None
        self._client = client or httpx.AsyncClient(
            base_url=settings.catalog_base_url,
            timeout=settings.request_timeout_seconds,
            headers=headers,
        )

    async def search(self, query: str, *, limit: int) -> tuple[MusicTrack, ...]:
        normalized = query.strip()
        if not normalized:
            raise ValueError("Music search query must not be empty")
        bounded_limit = min(max(limit, 1), self._settings.search_limit)
        payload = await self._get_json(
            "/search",
            params={"keywords": normalized, "type": 1, "limit": bounded_limit},
        )
        result = payload.get("result")
        songs = result.get("songs") if isinstance(result, Mapping) else None
        if not isinstance(songs, list):
            raise MusicCatalogError("Music catalog search response has no songs")
        tracks = tuple(
            track
            for item in songs[:bounded_limit]
            if (track := self._parse_track(item)) is not None
        )
        if not tracks:
            raise MusicNotFoundError("No music matched the query")
        return tracks

    async def resolve(self, track: MusicTrack) -> PlayableTrack:
        if track.source != "netease":
            raise MusicCatalogError("Unsupported music source")
        payload = await self._get_json(
            "/song/url",
            params={"id": track.source_id, "br": self._settings.bitrate},
        )
        values = payload.get("data")
        if not isinstance(values, list):
            raise MusicCatalogError("Music stream response has no data")
        for item in values:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("id", "")) != track.source_id:
                continue
            url = item.get("url")
            if isinstance(url, str) and url.strip():
                return PlayableTrack(track, url.strip())
        raise MusicNotFoundError("The selected music is not currently playable")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_json(
        self,
        path: str,
        *,
        params: Mapping[str, object],
    ) -> Mapping[str, Any]:
        try:
            response = await self._client.get(path, params=params)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MusicCatalogError("Music catalog request failed") from exc
        if not isinstance(payload, Mapping):
            raise MusicCatalogError("Music catalog returned invalid JSON")
        return payload

    @staticmethod
    def _parse_track(value: object) -> MusicTrack | None:
        if not isinstance(value, Mapping):
            return None
        source_id = str(value.get("id", "")).strip()
        title = value.get("name")
        if not source_id or not isinstance(title, str) or not title.strip():
            return None
        raw_artists = value.get("artists")
        if not isinstance(raw_artists, list):
            raw_artists = value.get("ar")
        artists: list[str] = []
        if isinstance(raw_artists, list):
            for artist in raw_artists:
                if isinstance(artist, Mapping):
                    name = artist.get("name")
                    if isinstance(name, str) and name.strip():
                        artists.append(name.strip())
        raw_duration = value.get("duration", value.get("dt"))
        duration_ms = raw_duration if isinstance(raw_duration, int) and raw_duration >= 0 else None
        return MusicTrack(
            source="netease",
            source_id=source_id,
            title=title.strip(),
            artists=tuple(artists),
            duration_ms=duration_ms,
        )
