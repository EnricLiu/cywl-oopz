"""Synchronous, one-request yt-dlp worker isolated from the asyncio process."""

from __future__ import annotations

import html
import importlib.metadata
import math
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from typing import Protocol
from urllib.parse import urlsplit

import yt_dlp

from .ytdlp_models import (
    YtDlpMode,
    YtDlpOperation,
    YtDlpWorkerError,
    YtDlpWorkerItem,
    YtDlpWorkerMedia,
    YtDlpWorkerRequest,
    YtDlpWorkerResponse,
)

_MAX_REQUEST_BYTES = 16 * 1024
_MAX_WARNINGS = 20
_MAX_WARNING_CHARACTERS = 500
_MAX_ERROR_CHARACTERS = 600
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_AUTHORIZATION = re.compile(r"(?i)(authorization|cookie|po[_ -]?token)\s*[:=]\s*\S+")
_VERSION = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


class _Extractor(Protocol):
    def extract_info(self, url: str, *, download: bool) -> object: ...


_ExtractorFactory = Callable[[dict[str, object]], AbstractContextManager[_Extractor]]


class _BoundedYtDlpLogger:
    """Capture useful warnings without allowing yt-dlp to write protocol stdout."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def debug(self, message: str) -> None:
        del message

    def info(self, message: str) -> None:
        del message

    def warning(self, message: str) -> None:
        self._append(message)

    def error(self, message: str) -> None:
        self._append(message)

    def _append(self, message: str) -> None:
        if len(self.warnings) >= _MAX_WARNINGS:
            return
        self.warnings.append(_safe_message(message, _MAX_WARNING_CHARACTERS))


class YtDlpWorker:
    """Map yt-dlp's unstable info dictionaries into the CYWL protocol."""

    def __init__(
        self,
        extractor_factory: _ExtractorFactory = yt_dlp.YoutubeDL,
    ) -> None:
        self._extractor_factory = extractor_factory

    def execute(self, request: YtDlpWorkerRequest) -> YtDlpWorkerResponse:
        logger = _BoundedYtDlpLogger()
        try:
            if request.operation is YtDlpOperation.PROBE:
                return self._probe(request)
            options = self._options(request, logger)
            with self._extractor_factory(options) as extractor:
                raw = extractor.extract_info(request.target, download=False)
            if not isinstance(raw, Mapping):
                raise _WorkerFailure("extractor_failed", "Extractor returned no metadata")
            items = tuple(self._item(value, request) for value in self._select_items(raw, request))
            if not items:
                raise _WorkerFailure("not_found", "No matching media was found")
            return YtDlpWorkerResponse(
                ok=True,
                items=items,
                warnings=tuple(logger.warnings),
            )
        except Exception as exc:
            failure = self._failure(exc)
            return YtDlpWorkerResponse(
                ok=False,
                warnings=tuple(logger.warnings),
                error=failure,
            )

    @staticmethod
    def _options(
        request: YtDlpWorkerRequest,
        logger: _BoundedYtDlpLogger,
    ) -> dict[str, object]:
        configuration = request.configuration
        options: dict[str, object] = {
            "quiet": True,
            "no_warnings": False,
            "noprogress": True,
            "skip_download": True,
            "socket_timeout": configuration.socket_timeout_seconds,
            "retries": 1,
            "extractor_retries": 1,
            "ignoreconfig": True,
            "cachedir": configuration.cache_dir,
            "logger": logger,
        }
        if request.mode is YtDlpMode.FLAT_SEARCH:
            options.update(
                {
                    "extract_flat": "in_playlist",
                    "playlistend": request.limit,
                    "noplaylist": False,
                }
            )
        else:
            options.update(
                {
                    "extract_flat": False,
                    "noplaylist": True,
                }
            )
        if request.mode is YtDlpMode.PLAYABLE_MEDIA:
            bitrate = configuration.max_audio_bitrate_kbps
            options["format"] = f"ba[abr<=?{bitrate}]/ba/b[height<=?480]/b"
        if request.profile == "youtube_public" or configuration.require_javascript:
            runtime_options: dict[str, object] = {}
            if configuration.js_runtime_path:
                runtime_options["path"] = configuration.js_runtime_path
            options["js_runtimes"] = {configuration.js_runtime: runtime_options}
        if request.profile == "youtube_public" and configuration.youtube_player_clients:
            options["extractor_args"] = {
                "youtube": {
                    "player_client": list(configuration.youtube_player_clients),
                }
            }
        if configuration.cookie_file:
            options["cookiefile"] = configuration.cookie_file
        return options

    @staticmethod
    def _select_items(
        root: Mapping[str, object],
        request: YtDlpWorkerRequest,
    ) -> tuple[Mapping[str, object], ...]:
        raw_entries = root.get("entries")
        if request.mode is YtDlpMode.FLAT_SEARCH:
            if not isinstance(raw_entries, list):
                raise _WorkerFailure("extractor_failed", "Search result has no entries")
            return tuple(item for item in raw_entries[: request.limit] if isinstance(item, Mapping))
        if isinstance(raw_entries, list):
            entries = tuple(item for item in raw_entries if isinstance(item, Mapping))
            if len(entries) != 1:
                raise _WorkerFailure(
                    "ambiguous_collection",
                    "Expected one media item but received a collection",
                )
            return entries
        return (root,)

    def _item(
        self,
        info: Mapping[str, object],
        request: YtDlpWorkerRequest,
    ) -> YtDlpWorkerItem:
        extractor_key = _first_text(info, "extractor_key", "ie_key", "extractor")
        if request.expected_extractors and extractor_key.casefold() not in {
            value.casefold() for value in request.expected_extractors
        }:
            raise _WorkerFailure(
                "unexpected_extractor",
                "The page resolved through an unexpected extractor",
            )
        identifier = _first_text(info, "id")
        title = _clean_text(_first_text(info, "title"), 512)
        artists = self._artists(info)
        duration = _finite_float(info.get("duration"))
        media = self._media(info) if request.mode is YtDlpMode.PLAYABLE_MEDIA else None
        webpage_url = _optional_clean_text(info.get("webpage_url"), 2_048)
        if webpage_url is None:
            webpage_url = _optional_clean_text(info.get("original_url"), 2_048)
        if webpage_url is None and request.mode is YtDlpMode.FLAT_SEARCH:
            # In flat-search mode ``url`` is an extractor page, not an expiring
            # media input. Providers need it to recover canonical platform IDs.
            webpage_url = _optional_clean_text(info.get("url"), 2_048)
        return YtDlpWorkerItem(
            extractor_key=extractor_key,
            id=identifier,
            title=title,
            artists=artists,
            channel=_optional_clean_text(info.get("channel"), 128),
            uploader=_optional_clean_text(info.get("uploader"), 128),
            duration_seconds=duration,
            webpage_url=webpage_url,
            live_status=_optional_clean_text(info.get("live_status"), 64),
            is_live=info.get("is_live") if isinstance(info.get("is_live"), bool) else None,
            availability=_optional_clean_text(info.get("availability"), 64),
            age_limit=_non_negative_integer(info.get("age_limit")),
            playlist_index=_non_negative_integer(info.get("playlist_index")),
            media=media,
        )

    @staticmethod
    def _artists(info: Mapping[str, object]) -> tuple[str, ...]:
        values: list[str] = []
        raw_artists = info.get("artists")
        if isinstance(raw_artists, list):
            for item in raw_artists:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, Mapping):
                    name = item.get("name")
                    if isinstance(name, str):
                        values.append(name)
        artist = info.get("artist")
        if not values and isinstance(artist, str):
            values.append(artist)
        return tuple(cleaned for item in values[:8] if (cleaned := _clean_text(item, 128)))

    @staticmethod
    def _media(root: Mapping[str, object]) -> YtDlpWorkerMedia:
        selected = root
        if not _usable_media_mapping(selected):
            selected = _single_requested_format(root, "requested_downloads")
        if not _usable_media_mapping(selected):
            selected = _single_requested_format(root, "requested_formats")
        if not _usable_media_mapping(selected):
            raise _WorkerFailure("no_audio", "Extractor returned no single audio input")
        if selected.get("has_drm") is True or root.get("has_drm") is True:
            raise _WorkerFailure("drm", "DRM protected media is not supported")
        url = _first_text(selected, "url")
        if urlsplit(url).scheme.casefold() not in {"http", "https"}:
            raise _WorkerFailure("unsupported_protocol", "Media URL is not HTTP or HTTPS")
        audio_codec = _optional_clean_text(selected.get("acodec"), 64)
        if audio_codec is None or audio_codec.casefold() == "none":
            raise _WorkerFailure("no_audio", "Selected media format has no audio")
        headers: dict[str, str] = {}
        for source in (root.get("http_headers"), selected.get("http_headers")):
            if isinstance(source, Mapping):
                for name, value in source.items():
                    if isinstance(name, str) and isinstance(value, str):
                        headers[name] = value
        return YtDlpWorkerMedia(
            url=url,
            http_headers=tuple(headers.items()),
            protocol=_optional_clean_text(selected.get("protocol"), 64),
            format_id=_optional_clean_text(selected.get("format_id"), 128),
            container=_optional_clean_text(selected.get("ext"), 32),
            audio_codec=audio_codec,
            video_codec=_optional_clean_text(selected.get("vcodec"), 64),
        )

    @staticmethod
    def _probe(request: YtDlpWorkerRequest) -> YtDlpWorkerResponse:
        capabilities = {"yt_dlp": importlib.metadata.version("yt-dlp")}
        try:
            capabilities["yt_dlp_ejs"] = importlib.metadata.version("yt-dlp-ejs")
        except importlib.metadata.PackageNotFoundError:
            if request.configuration.require_javascript:
                raise _WorkerFailure(
                    "javascript_support_missing",
                    "yt-dlp-ejs is required for YouTube extraction",
                ) from None
        if request.configuration.require_javascript:
            runtime = request.configuration.js_runtime
            configured = request.configuration.js_runtime_path
            executable = configured or shutil.which(runtime)
            if not executable:
                raise _WorkerFailure(
                    "javascript_runtime_missing",
                    f"Configured JavaScript runtime is unavailable: {runtime}",
                )
            try:
                completed = subprocess.run(
                    (executable, "--version"),
                    capture_output=True,
                    check=False,
                    timeout=2,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise _WorkerFailure(
                    "javascript_runtime_missing",
                    f"Could not execute JavaScript runtime: {type(exc).__name__}",
                ) from exc
            output = (completed.stdout or completed.stderr).decode(errors="replace")
            if completed.returncode != 0:
                raise _WorkerFailure(
                    "javascript_runtime_missing",
                    "JavaScript runtime capability check failed",
                )
            version = _parse_version(output)
            minimum = (2, 3, 0) if runtime == "deno" else (22, 0, 0)
            if version < minimum:
                raise _WorkerFailure(
                    "javascript_runtime_outdated",
                    f"JavaScript runtime {runtime} is below the required version",
                )
            capabilities[runtime] = ".".join(str(item) for item in version)
        return YtDlpWorkerResponse(ok=True, capabilities=capabilities)

    @staticmethod
    def _failure(exc: Exception) -> YtDlpWorkerError:
        if isinstance(exc, _WorkerFailure):
            return YtDlpWorkerError(exc.code, exc.message, type(exc).__name__)
        name = type(exc).__name__
        message = str(exc)
        normalized = message.casefold()
        if name == "GeoRestrictedError" or "not available in your country" in normalized:
            code = "geo_restricted"
        elif name == "UnsupportedError":
            code = "unsupported_url"
        elif "sign in" in normalized or "login" in normalized or "authentication" in normalized:
            code = "login_required"
        elif (
            "rate limit" in normalized
            or "too many requests" in normalized
            or "http error 429" in normalized
            or "http error 412" in normalized
            or "precondition failed" in normalized
        ):
            code = "rate_limited"
        elif "not found" in normalized or "unavailable" in normalized:
            code = "not_found"
        else:
            code = "extractor_failed"
        return YtDlpWorkerError(code, _safe_message(message, _MAX_ERROR_CHARACTERS), name)


class _WorkerFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _usable_media_mapping(value: Mapping[str, object]) -> bool:
    return isinstance(value.get("url"), str) and value.get("acodec") not in {None, "none"}


def _single_requested_format(
    root: Mapping[str, object],
    name: str,
) -> Mapping[str, object]:
    values = root.get(name)
    if not isinstance(values, list):
        return {}
    mappings = tuple(item for item in values if isinstance(item, Mapping))
    if len(mappings) > 1:
        raise _WorkerFailure(
            "multiple_formats_unsupported",
            "Selected media requires multiple inputs",
        )
    return mappings[0] if mappings else {}


def _first_text(values: Mapping[str, object], *names: str) -> str:
    for name in names:
        value = values.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise _WorkerFailure("extractor_failed", f"Extractor result has no {names[0]}")


def _clean_text(value: str, limit: int) -> str:
    normalized = _CONTROL_CHARACTERS.sub(" ", html.unescape(value))
    return " ".join(normalized.split())[:limit]


def _optional_clean_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _clean_text(value, limit)
    return cleaned or None


def _finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted >= 0 else None


def _non_negative_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _safe_message(value: str, limit: int) -> str:
    redacted = _URL.sub("<url>", value)
    redacted = _AUTHORIZATION.sub(r"\1=<redacted>", redacted)
    return _clean_text(redacted, limit) or "External media extraction failed"


def _parse_version(value: str) -> tuple[int, int, int]:
    match = _VERSION.search(value)
    if match is None:
        raise _WorkerFailure(
            "javascript_runtime_outdated",
            "Could not parse JavaScript runtime version",
        )
    return tuple(int(item or 0) for item in match.groups())


def main() -> int:
    raw = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if len(raw) > _MAX_REQUEST_BYTES:
        response = YtDlpWorkerResponse(
            ok=False,
            error=YtDlpWorkerError("invalid_request", "Worker request is too large"),
        )
    else:
        try:
            request = YtDlpWorkerRequest.from_bytes(raw)
            response = YtDlpWorker().execute(request)
        except Exception as exc:
            response = YtDlpWorkerResponse(
                ok=False,
                error=YtDlpWorkerError(
                    "invalid_request",
                    _safe_message(str(exc), _MAX_ERROR_CHARACTERS),
                    type(exc).__name__,
                ),
            )
    sys.stdout.buffer.write(response.to_bytes())
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
