"""Versioned JSON boundary between the asyncio application and yt-dlp worker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

PROTOCOL_VERSION = 1


class YtDlpOperation(StrEnum):
    SEARCH = "search"
    LOOKUP = "lookup"
    INSPECT = "inspect"
    RESOLVE = "resolve"
    PROBE = "probe"


class YtDlpMode(StrEnum):
    FLAT_SEARCH = "flat_search"
    FULL_METADATA = "full_metadata"
    PLAYABLE_MEDIA = "playable_media"
    RUNTIME_ONLY = "runtime_only"


@dataclass(frozen=True, slots=True)
class YtDlpWorkerConfiguration:
    socket_timeout_seconds: float = 8.0
    max_audio_bitrate_kbps: int = 192
    cache_dir: str = ".cache/yt-dlp"
    js_runtime: str = "deno"
    js_runtime_path: str = ""
    cookie_file: str = field(default="", repr=False)
    require_javascript: bool = False

    def __post_init__(self) -> None:
        if self.socket_timeout_seconds <= 0:
            raise ValueError("yt-dlp socket timeout must be positive")
        if not 1 <= self.max_audio_bitrate_kbps <= 512:
            raise ValueError("yt-dlp audio bitrate must be between 1 and 512 kbps")
        if not self.cache_dir.strip():
            raise ValueError("yt-dlp cache directory must not be empty")
        if self.js_runtime not in {"deno", "node"}:
            raise ValueError("yt-dlp JavaScript runtime must be deno or node")

    def to_payload(self) -> dict[str, object]:
        return {
            "socket_timeout_seconds": self.socket_timeout_seconds,
            "max_audio_bitrate_kbps": self.max_audio_bitrate_kbps,
            "cache_dir": self.cache_dir,
            "js_runtime": self.js_runtime,
            "js_runtime_path": self.js_runtime_path,
            "cookie_file": self.cookie_file,
            "require_javascript": self.require_javascript,
        }

    @classmethod
    def from_payload(cls, value: object) -> YtDlpWorkerConfiguration:
        if not isinstance(value, dict):
            raise ValueError("yt-dlp worker configuration must be an object")
        return cls(
            socket_timeout_seconds=_float(value, "socket_timeout_seconds"),
            max_audio_bitrate_kbps=_integer(value, "max_audio_bitrate_kbps"),
            cache_dir=_text(value, "cache_dir"),
            js_runtime=_text(value, "js_runtime"),
            js_runtime_path=_optional_text(value, "js_runtime_path"),
            cookie_file=_optional_text(value, "cookie_file"),
            require_javascript=_boolean(value, "require_javascript"),
        )


@dataclass(frozen=True, slots=True)
class YtDlpWorkerRequest:
    operation: YtDlpOperation
    mode: YtDlpMode
    target: str = field(default="", repr=False)
    limit: int = 1
    expected_extractors: tuple[str, ...] = ()
    profile: str = "generic"
    configuration: YtDlpWorkerConfiguration = field(
        default_factory=YtDlpWorkerConfiguration,
        repr=False,
    )
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", YtDlpOperation(self.operation))
        object.__setattr__(self, "mode", YtDlpMode(self.mode))
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unsupported yt-dlp worker protocol version")
        if self.operation is not YtDlpOperation.PROBE and not self.target.strip():
            raise ValueError("yt-dlp worker target must not be empty")
        if not 1 <= self.limit <= 100:
            raise ValueError("yt-dlp worker limit must be between 1 and 100")
        if not self.profile.strip():
            raise ValueError("yt-dlp worker profile must not be empty")

    def to_bytes(self) -> bytes:
        payload = {
            "protocol_version": self.protocol_version,
            "operation": self.operation.value,
            "mode": self.mode.value,
            "target": self.target,
            "limit": self.limit,
            "expected_extractors": list(self.expected_extractors),
            "profile": self.profile,
            "configuration": self.configuration.to_payload(),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, value: bytes) -> YtDlpWorkerRequest:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("yt-dlp worker request is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("yt-dlp worker request must be an object")
        raw_extractors = payload.get("expected_extractors", [])
        if not isinstance(raw_extractors, list) or not all(
            isinstance(item, str) and item for item in raw_extractors
        ):
            raise ValueError("yt-dlp expected extractors must be text")
        return cls(
            protocol_version=_integer(payload, "protocol_version"),
            operation=YtDlpOperation(_text(payload, "operation")),
            mode=YtDlpMode(_text(payload, "mode")),
            target=_optional_text(payload, "target"),
            limit=_integer(payload, "limit"),
            expected_extractors=tuple(raw_extractors),
            profile=_text(payload, "profile"),
            configuration=YtDlpWorkerConfiguration.from_payload(payload.get("configuration")),
        )


@dataclass(frozen=True, slots=True)
class YtDlpWorkerMedia:
    url: str = field(repr=False)
    http_headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    protocol: str | None = None
    format_id: str | None = None
    container: str | None = None
    audio_codec: str | None = None
    video_codec: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "url": self.url,
            "http_headers": [list(item) for item in self.http_headers],
            "protocol": self.protocol,
            "format_id": self.format_id,
            "container": self.container,
            "audio_codec": self.audio_codec,
            "video_codec": self.video_codec,
        }

    @classmethod
    def from_payload(cls, value: object) -> YtDlpWorkerMedia:
        if not isinstance(value, dict):
            raise ValueError("yt-dlp media must be an object")
        raw_headers = value.get("http_headers", [])
        if not isinstance(raw_headers, list):
            raise ValueError("yt-dlp media headers must be a list")
        headers: list[tuple[str, str]] = []
        for item in raw_headers:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(part, str) for part in item)
            ):
                raise ValueError("yt-dlp media header is invalid")
            headers.append((item[0], item[1]))
        return cls(
            url=_text(value, "url"),
            http_headers=tuple(headers),
            protocol=_nullable_text(value, "protocol"),
            format_id=_nullable_text(value, "format_id"),
            container=_nullable_text(value, "container"),
            audio_codec=_nullable_text(value, "audio_codec"),
            video_codec=_nullable_text(value, "video_codec"),
        )


@dataclass(frozen=True, slots=True)
class YtDlpWorkerItem:
    extractor_key: str
    id: str
    title: str
    artists: tuple[str, ...] = ()
    channel: str | None = None
    uploader: str | None = None
    duration_seconds: float | None = None
    webpage_url: str | None = None
    live_status: str | None = None
    is_live: bool | None = None
    availability: str | None = None
    age_limit: int | None = None
    playlist_index: int | None = None
    media: YtDlpWorkerMedia | None = field(default=None, repr=False)

    def to_payload(self) -> dict[str, object]:
        return {
            "extractor_key": self.extractor_key,
            "id": self.id,
            "title": self.title,
            "artists": list(self.artists),
            "channel": self.channel,
            "uploader": self.uploader,
            "duration_seconds": self.duration_seconds,
            "webpage_url": self.webpage_url,
            "live_status": self.live_status,
            "is_live": self.is_live,
            "availability": self.availability,
            "age_limit": self.age_limit,
            "playlist_index": self.playlist_index,
            "media": self.media.to_payload() if self.media is not None else None,
        }

    @classmethod
    def from_payload(cls, value: object) -> YtDlpWorkerItem:
        if not isinstance(value, dict):
            raise ValueError("yt-dlp item must be an object")
        raw_artists = value.get("artists", [])
        if not isinstance(raw_artists, list) or not all(
            isinstance(item, str) for item in raw_artists
        ):
            raise ValueError("yt-dlp item artists must be text")
        raw_media = value.get("media")
        return cls(
            extractor_key=_text(value, "extractor_key"),
            id=_text(value, "id"),
            title=_text(value, "title"),
            artists=tuple(raw_artists),
            channel=_nullable_text(value, "channel"),
            uploader=_nullable_text(value, "uploader"),
            duration_seconds=_nullable_float(value, "duration_seconds"),
            webpage_url=_nullable_text(value, "webpage_url"),
            live_status=_nullable_text(value, "live_status"),
            is_live=_nullable_boolean(value, "is_live"),
            availability=_nullable_text(value, "availability"),
            age_limit=_nullable_integer(value, "age_limit"),
            playlist_index=_nullable_integer(value, "playlist_index"),
            media=(YtDlpWorkerMedia.from_payload(raw_media) if raw_media is not None else None),
        )


@dataclass(frozen=True, slots=True)
class YtDlpWorkerError:
    code: str
    message: str
    upstream_type: str = ""

    def to_payload(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "upstream_type": self.upstream_type,
        }


@dataclass(frozen=True, slots=True)
class YtDlpWorkerResponse:
    ok: bool
    items: tuple[YtDlpWorkerItem, ...] = ()
    warnings: tuple[str, ...] = ()
    error: YtDlpWorkerError | None = None
    capabilities: dict[str, str] = field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unsupported yt-dlp worker protocol version")
        if self.ok == (self.error is not None):
            raise ValueError("yt-dlp response success and error fields disagree")

    def to_bytes(self) -> bytes:
        payload = {
            "protocol_version": self.protocol_version,
            "ok": self.ok,
            "items": [item.to_payload() for item in self.items],
            "warnings": list(self.warnings),
            "error": self.error.to_payload() if self.error is not None else None,
            "capabilities": self.capabilities,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, value: bytes) -> YtDlpWorkerResponse:
        try:
            payload = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("yt-dlp worker response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("yt-dlp worker response must be an object")
        raw_items = payload.get("items", [])
        raw_warnings = payload.get("warnings", [])
        raw_capabilities = payload.get("capabilities", {})
        if not isinstance(raw_items, list):
            raise ValueError("yt-dlp response items must be a list")
        if not isinstance(raw_warnings, list) or not all(
            isinstance(item, str) for item in raw_warnings
        ):
            raise ValueError("yt-dlp response warnings must be text")
        if not isinstance(raw_capabilities, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in raw_capabilities.items()
        ):
            raise ValueError("yt-dlp response capabilities must be text")
        raw_error = payload.get("error")
        error = None
        if raw_error is not None:
            if not isinstance(raw_error, dict):
                raise ValueError("yt-dlp response error must be an object")
            error = YtDlpWorkerError(
                _text(raw_error, "code"),
                _text(raw_error, "message"),
                _optional_text(raw_error, "upstream_type"),
            )
        return cls(
            protocol_version=_integer(payload, "protocol_version"),
            ok=_boolean(payload, "ok"),
            items=tuple(YtDlpWorkerItem.from_payload(item) for item in raw_items),
            warnings=tuple(raw_warnings),
            error=error,
            capabilities=dict(raw_capabilities),
        )


def _text(values: dict[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _optional_text(values: dict[str, Any], name: str) -> str:
    value = values.get(name, "")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    return value


def _nullable_text(values: dict[str, Any], name: str) -> str | None:
    value = values.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be text or null")
    return value


def _integer(values: dict[str, Any], name: str) -> int:
    value = values.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _nullable_integer(values: dict[str, Any], name: str) -> int | None:
    value = values.get(name)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{name} must be an integer or null")
    return value


def _float(values: dict[str, Any], name: str) -> float:
    value = values.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _nullable_float(values: dict[str, Any], name: str) -> float | None:
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number or null")
    return float(value)


def _boolean(values: dict[str, Any], name: str) -> bool:
    value = values.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _nullable_boolean(values: dict[str, Any], name: str) -> bool | None:
    value = values.get(name)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean or null")
    return value
