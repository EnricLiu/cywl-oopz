"""Async NeteaseCloudMusicApi-compatible catalog adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from cywl_oopz.settings import MusicSettings

from .errors import (
    MusicCatalogError,
    MusicNotFoundError,
    NeteasePlaylistNotFoundError,
    NeteasePlaylistReferenceError,
)
from .models import MusicTrack, NeteasePlaylistSnapshot, PlayableTrack


@dataclass(frozen=True, slots=True)
class NeteasePlaylistReference:
    """One validated numeric playlist ID parsed from an ID or canonical URL."""

    playlist_id: str

    @classmethod
    def parse(cls, value: str) -> NeteasePlaylistReference:
        normalized = value.strip()
        if normalized.isdigit() and len(normalized) <= 20:
            return cls(normalized)
        parsed = urlparse(normalized)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme.casefold() not in {"http", "https"} or (
            host != "music.163.com" and not host.endswith(".music.163.com")
        ):
            raise NeteasePlaylistReferenceError(
                "Netease playlist must be a numeric ID or music.163.com URL"
            )
        candidates = [parse_qs(parsed.query).get("id", ())]
        fragment_query = parsed.fragment.partition("?")[2]
        if fragment_query:
            candidates.append(parse_qs(fragment_query).get("id", ()))
        for values in candidates:
            if values and values[0].isdigit() and len(values[0]) <= 20:
                return cls(values[0])
        raise NeteasePlaylistReferenceError("Netease playlist URL has no numeric playlist ID")


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
        catalog_host = (urlparse(settings.catalog_base_url).hostname or "").casefold()
        self._client = client or httpx.AsyncClient(
            base_url=settings.catalog_base_url,
            timeout=settings.request_timeout_seconds,
            headers=headers,
            trust_env=catalog_host not in {"127.0.0.1", "::1", "localhost"},
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

    async def playlist(
        self,
        reference: str,
        *,
        limit: int,
    ) -> NeteasePlaylistSnapshot:
        """Load playlist metadata plus a bounded slice from the complete-track endpoint."""
        if limit <= 0:
            raise ValueError("Netease playlist track limit must be positive")
        playlist_id = NeteasePlaylistReference.parse(reference).playlist_id
        detail_payload = await self._get_json(
            "/playlist/detail",
            params={"id": playlist_id, "s": 0},
        )
        playlist = detail_payload.get("playlist")
        if not isinstance(playlist, Mapping):
            raise NeteasePlaylistNotFoundError("Netease playlist was not found")
        name = playlist.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MusicCatalogError("Netease playlist response has no name")
        declared_track_count = self._playlist_track_count(playlist)

        tracks_payload = await self._get_json(
            "/playlist/track/all",
            params={"id": playlist_id, "limit": limit, "offset": 0},
        )
        songs = tracks_payload.get("songs")
        if not isinstance(songs, list):
            raise MusicCatalogError("Netease playlist track response has no songs")
        tracks = tuple(
            track for item in songs[:limit] if (track := self._parse_track(item)) is not None
        )
        return NeteasePlaylistSnapshot(
            playlist_id,
            name.strip(),
            max(declared_track_count, len(tracks)),
            tracks,
        )

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

    @staticmethod
    def _playlist_track_count(playlist: Mapping[object, object]) -> int:
        raw_count = playlist.get("trackCount")
        if isinstance(raw_count, int) and raw_count >= 0:
            return raw_count
        track_ids = playlist.get("trackIds")
        return len(track_ids) if isinstance(track_ids, list) else 0
