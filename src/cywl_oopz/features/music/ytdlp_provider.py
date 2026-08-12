"""Shared policy and error translation for yt-dlp-backed music providers."""

from __future__ import annotations

import logging
from abc import ABC

from cywl_oopz.integrations.media.ytdlp_models import (
    YtDlpMode,
    YtDlpOperation,
    YtDlpWorkerItem,
    YtDlpWorkerRequest,
    YtDlpWorkerResponse,
)
from cywl_oopz.integrations.media.ytdlp_runner import YtDlpProcessRunner
from cywl_oopz.settings import MusicSettings

from .errors import (
    MusicAuthenticationRequiredError,
    MusicGeoRestrictedError,
    MusicLiveUnsupportedError,
    MusicNoAudioFormatError,
    MusicNotFoundError,
    MusicReferenceError,
    MusicSourceRateLimitedError,
    MusicSourceUnavailableError,
    MusicTrackTooLongError,
    MusicUnsupportedContentError,
)
from .models import MusicProviderHealth, MusicProviderHealthState, MusicSourceKind

logger = logging.getLogger(__name__)

_ALLOWED_NON_LIVE_STATES = frozenset({"not_live", "was_live"})


class YtDlpMusicProviderBase(ABC):
    """Own the stable project boundary around one shared extractor runner."""

    source: MusicSourceKind
    profile: str
    expected_extractors: tuple[str, ...]

    def __init__(
        self,
        settings: MusicSettings,
        runner: YtDlpProcessRunner,
        *,
        cookie_file: str,
        require_javascript: bool,
    ) -> None:
        self._settings = settings
        self._runner = runner
        self._cookie_file = cookie_file
        self._require_javascript = require_javascript

    async def _extract(
        self,
        operation: YtDlpOperation,
        mode: YtDlpMode,
        target: str,
        *,
        limit: int = 1,
    ) -> tuple[YtDlpWorkerItem, ...]:
        response = await self._runner.run(
            YtDlpWorkerRequest(
                operation=operation,
                mode=mode,
                target=target,
                limit=limit,
                expected_extractors=self.expected_extractors,
                profile=self.profile,
                configuration=self._runner.configuration(
                    cookie_file=self._cookie_file,
                    require_javascript=self._require_javascript,
                ),
            )
        )
        self._raise_worker_error(response)
        if not response.items:
            raise MusicNotFoundError(f"{self.source.value} returned no media")
        return response.items

    def _validate_content(
        self,
        item: YtDlpWorkerItem,
        *,
        require_duration: bool,
    ) -> int | None:
        if (item.availability or "").casefold() in {
            "private",
            "premium_only",
            "subscriber_only",
            "needs_auth",
        }:
            raise MusicAuthenticationRequiredError("Media requires source authentication")
        live_status = (item.live_status or "").casefold()
        if item.is_live is True or (live_status and live_status not in _ALLOWED_NON_LIVE_STATES):
            raise MusicLiveUnsupportedError("Live and upcoming media are not supported")
        if item.duration_seconds is None:
            if require_duration:
                raise MusicUnsupportedContentError("Media duration is unavailable")
            return None
        if item.duration_seconds > self._settings.max_track_duration_seconds:
            raise MusicTrackTooLongError("Media exceeds the configured maximum playback duration")
        return round(item.duration_seconds * 1_000)

    async def health(self) -> MusicProviderHealth:
        """Check local extractor capability without making a source request."""
        try:
            response = await self._runner.probe(require_javascript=self._require_javascript)
        except MusicSourceUnavailableError as exc:
            logger.warning(
                "Music provider capability probe failed: source=%s error=%s",
                self.source.value,
                type(exc).__name__,
            )
            return MusicProviderHealth(
                self.source,
                MusicProviderHealthState.UNAVAILABLE,
                type(exc).__name__,
            )
        if not response.ok:
            assert response.error is not None
            return MusicProviderHealth(
                self.source,
                MusicProviderHealthState.UNAVAILABLE,
                response.error.code,
            )
        return MusicProviderHealth(self.source, MusicProviderHealthState.READY)

    async def aclose(self) -> None:
        """The composition root owns and closes the shared runner."""

    @staticmethod
    def _raise_worker_error(response: YtDlpWorkerResponse) -> None:
        if response.ok:
            return
        assert response.error is not None
        code = response.error.code
        message = response.error.message
        if code == "not_found":
            raise MusicNotFoundError(message)
        if code in {"unsupported_url", "invalid_request"}:
            raise MusicReferenceError(message)
        if code == "login_required":
            raise MusicAuthenticationRequiredError(message)
        if code == "geo_restricted":
            raise MusicGeoRestrictedError(message)
        if code == "rate_limited":
            raise MusicSourceRateLimitedError(message)
        if code == "no_audio":
            raise MusicNoAudioFormatError(message)
        if code == "live_unsupported":
            raise MusicLiveUnsupportedError(message)
        if code == "duration_limit":
            raise MusicTrackTooLongError(message)
        if code in {
            "ambiguous_collection",
            "drm",
            "multiple_formats_unsupported",
            "unexpected_extractor",
            "unsupported_protocol",
        }:
            raise MusicUnsupportedContentError(message)
        raise MusicSourceUnavailableError(message)
