from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

import cywl_oopz.integrations.media.ytdlp_worker as worker_module
from cywl_oopz.integrations.media.ytdlp_models import (
    YtDlpMode,
    YtDlpOperation,
    YtDlpWorkerConfiguration,
    YtDlpWorkerRequest,
    YtDlpWorkerResponse,
)
from cywl_oopz.integrations.media.ytdlp_worker import YtDlpWorker


class FakeExtractor:
    def __init__(self, payload: object, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def __enter__(self) -> FakeExtractor:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def extract_info(self, url: str, *, download: bool) -> object:
        assert url
        assert download is False
        if self._error is not None:
            raise self._error
        return self._payload


class FakeExtractorFactory:
    def __init__(self, payload: object, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.options: list[dict[str, object]] = []

    def __call__(self, options: dict[str, object]) -> FakeExtractor:
        self.options.append(options)
        return FakeExtractor(self.payload, self.error)


def request(
    operation: YtDlpOperation,
    mode: YtDlpMode,
    *,
    expected: tuple[str, ...] = ("Youtube",),
) -> YtDlpWorkerRequest:
    return YtDlpWorkerRequest(
        operation=operation,
        mode=mode,
        target="ytsearch2:初音未来",
        limit=2,
        expected_extractors=expected,
        profile="youtube_public",
        configuration=YtDlpWorkerConfiguration(
            socket_timeout_seconds=7,
            max_audio_bitrate_kbps=160,
            cache_dir="/tmp/cywl-ytdlp-worker-test",
            js_runtime="node",
            js_runtime_path="/fixture/node",
            youtube_player_clients=("mweb",),
        ),
    )


def test_ytdlp_worker_maps_flat_search_without_leaking_media() -> None:
    factory = FakeExtractorFactory(
        {
            "_type": "playlist",
            "entries": [
                {
                    "extractor_key": "Youtube",
                    "id": "abcdefghijk",
                    "title": "Tell &amp; Your\x00 World",
                    "channel": "Miku Channel",
                    "duration": 245.25,
                    "webpage_url": "https://www.youtube.com/watch?v=abcdefghijk",
                    "url": "https://signed.example/should-not-leak",
                    "acodec": "opus",
                }
            ],
        }
    )
    worker = YtDlpWorker(factory)

    response = worker.execute(request(YtDlpOperation.SEARCH, YtDlpMode.FLAT_SEARCH))

    assert response.ok is True
    assert len(response.items) == 1
    assert response.items[0].title == "Tell & Your World"
    assert response.items[0].media is None
    assert factory.options[0]["extract_flat"] == "in_playlist"
    assert factory.options[0]["playlistend"] == 2
    assert factory.options[0]["socket_timeout"] == 7
    assert factory.options[0]["js_runtimes"] == {"node": {"path": "/fixture/node"}}
    assert factory.options[0]["extractor_args"] == {"youtube": {"player_client": ["mweb"]}}


def test_ytdlp_worker_retains_flat_page_url_when_webpage_url_is_absent() -> None:
    factory = FakeExtractorFactory(
        {
            "_type": "playlist",
            "entries": [
                {
                    "ie_key": "BiliBili",
                    "id": "170001",
                    "title": "fixture",
                    "url": "https://www.bilibili.com/video/BV17x411w7KC",
                }
            ],
        }
    )

    response = YtDlpWorker(factory).execute(
        request(
            YtDlpOperation.SEARCH,
            YtDlpMode.FLAT_SEARCH,
            expected=("BiliBili",),
        )
    )

    assert response.ok is True
    assert response.items[0].webpage_url == ("https://www.bilibili.com/video/BV17x411w7KC")


def test_ytdlp_worker_selects_one_audio_input_and_merges_headers() -> None:
    factory = FakeExtractorFactory(
        {
            "extractor_key": "Youtube",
            "id": "abcdefghijk",
            "title": "Tell Your World",
            "artist": "初音未来",
            "duration": 245,
            "is_live": False,
            "live_status": "not_live",
            "http_headers": {
                "Referer": "https://www.youtube.com/",
                "User-Agent": "root-agent",
            },
            "requested_downloads": [
                {
                    "url": "https://media.example/audio?token=sensitive",
                    "protocol": "https",
                    "format_id": "251",
                    "ext": "webm",
                    "acodec": "opus",
                    "vcodec": "none",
                    "http_headers": {"User-Agent": "format-agent"},
                }
            ],
        }
    )

    response = YtDlpWorker(factory).execute(
        request(YtDlpOperation.RESOLVE, YtDlpMode.PLAYABLE_MEDIA)
    )

    assert response.ok is True
    item = response.items[0]
    assert item.artists == ("初音未来",)
    assert item.is_live is False
    assert item.media is not None
    assert item.media.format_id == "251"
    assert item.media.audio_codec == "opus"
    assert dict(item.media.http_headers) == {
        "Referer": "https://www.youtube.com/",
        "User-Agent": "format-agent",
    }
    assert factory.options[0]["format"] == "ba[abr<=?160]/ba/b[height<=?480]/b"


def test_ytdlp_worker_rejects_collections_and_multi_input_formats() -> None:
    collection = FakeExtractorFactory(
        {
            "entries": [
                {"extractor_key": "Youtube", "id": "one", "title": "one"},
                {"extractor_key": "Youtube", "id": "two", "title": "two"},
            ]
        }
    )
    collection_response = YtDlpWorker(collection).execute(
        request(YtDlpOperation.LOOKUP, YtDlpMode.FULL_METADATA)
    )
    assert collection_response.ok is False
    assert collection_response.error is not None
    assert collection_response.error.code == "ambiguous_collection"

    multi_format = FakeExtractorFactory(
        {
            "extractor_key": "Youtube",
            "id": "abcdefghijk",
            "title": "fixture",
            "requested_formats": [
                {"url": "https://media.example/video", "acodec": "none"},
                {"url": "https://media.example/audio", "acodec": "opus"},
            ],
        }
    )
    format_response = YtDlpWorker(multi_format).execute(
        request(YtDlpOperation.RESOLVE, YtDlpMode.PLAYABLE_MEDIA)
    )
    assert format_response.ok is False
    assert format_response.error is not None
    assert format_response.error.code == "multiple_formats_unsupported"


def test_ytdlp_worker_bounds_and_redacts_external_errors() -> None:
    secret_url = "https://media.example/audio?token=do-not-return"
    factory = FakeExtractorFactory(
        {},
        RuntimeError(f"HTTP error at {secret_url} Cookie: session-secret"),
    )

    response = YtDlpWorker(factory).execute(request(YtDlpOperation.LOOKUP, YtDlpMode.FULL_METADATA))

    assert response.ok is False
    assert response.error is not None
    assert secret_url not in response.error.message
    assert "session-secret" not in response.error.message
    assert "<url>" in response.error.message


def test_ytdlp_worker_classifies_bilibili_412_as_rate_limited() -> None:
    response = YtDlpWorker(
        FakeExtractorFactory({}, RuntimeError("HTTP Error 412: Precondition Failed"))
    ).execute(
        request(
            YtDlpOperation.SEARCH,
            YtDlpMode.FLAT_SEARCH,
            expected=("BiliBili",),
        )
    )

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "rate_limited"


def test_ytdlp_protocol_round_trip_preserves_only_owned_fields() -> None:
    original = request(YtDlpOperation.SEARCH, YtDlpMode.FLAT_SEARCH)
    decoded = YtDlpWorkerRequest.from_bytes(original.to_bytes())
    assert decoded == original
    assert "初音未来" not in repr(decoded)
    assert "/fixture/node" not in repr(decoded)

    response = YtDlpWorkerResponse(ok=True, capabilities={"yt_dlp": "2026.7.4"})
    assert YtDlpWorkerResponse.from_bytes(response.to_bytes()) == response


def test_ytdlp_worker_probe_validates_javascript_runtime_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"yt-dlp": "2026.7.4", "yt-dlp-ejs": "0.8.0"}
    monkeypatch.setattr(worker_module.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(worker_module.shutil, "which", lambda name: f"/opt/{name}")
    monkeypatch.setattr(
        worker_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=b"deno 2.3.1\n",
            stderr=b"",
            returncode=0,
        ),
    )
    probe = YtDlpWorkerRequest(
        operation=YtDlpOperation.PROBE,
        mode=YtDlpMode.RUNTIME_ONLY,
        profile="youtube_public",
        configuration=YtDlpWorkerConfiguration(require_javascript=True),
    )

    response = YtDlpWorker().execute(probe)

    assert response.ok is True
    assert response.capabilities == {
        "yt_dlp": "2026.7.4",
        "yt_dlp_ejs": "0.8.0",
        "deno": "2.3.1",
    }


def test_ytdlp_worker_probe_reports_missing_javascript_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"yt-dlp": "2026.7.4", "yt-dlp-ejs": "0.8.0"}
    monkeypatch.setattr(worker_module.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(worker_module.shutil, "which", lambda _name: None)
    probe = YtDlpWorkerRequest(
        operation=YtDlpOperation.PROBE,
        mode=YtDlpMode.RUNTIME_ONLY,
        profile="youtube_public",
        configuration=YtDlpWorkerConfiguration(require_javascript=True),
    )

    response = YtDlpWorker().execute(probe)

    assert response.ok is False
    assert response.error is not None
    assert response.error.code == "javascript_runtime_missing"


def test_worker_fixture_payloads_remain_plain_mappings() -> None:
    """Guard fake fixtures from accidentally depending on yt-dlp concrete dict types."""
    payload: Mapping[str, object] = {"id": "fixture"}
    assert payload["id"] == "fixture"
