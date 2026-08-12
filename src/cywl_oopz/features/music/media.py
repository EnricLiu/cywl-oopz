"""Validation policy for provider-supplied media transport metadata."""

from __future__ import annotations

from collections.abc import Iterable

_CANONICAL_HEADER_NAMES = {
    "user-agent": "User-Agent",
    "referer": "Referer",
    "origin": "Origin",
    "accept": "Accept",
    "accept-language": "Accept-Language",
}
_HEADER_ORDER = tuple(_CANONICAL_HEADER_NAMES)


class MusicMediaHeaderPolicy:
    """Reduce extractor headers to a bounded FFmpeg-safe allow-list."""

    def __init__(
        self,
        *,
        max_header_value_characters: int = 2_048,
        max_total_characters: int = 8_192,
    ) -> None:
        if max_header_value_characters <= 0 or max_total_characters <= 0:
            raise ValueError("Music media header limits must be positive")
        self._max_header_value_characters = max_header_value_characters
        self._max_total_characters = max_total_characters

    def sanitize(self, values: Iterable[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        """Validate every pair, drop non-allowed fields, and make ordering stable."""
        accepted: dict[str, str] = {}
        for raw_name, raw_value in values:
            if not isinstance(raw_name, str) or not isinstance(raw_value, str):
                raise ValueError("Music media headers must contain text pairs")
            if self._has_forbidden_characters(raw_name) or self._has_forbidden_characters(
                raw_value
            ):
                raise ValueError("Music media headers must not contain control characters")
            normalized_name = raw_name.strip().casefold()
            value = raw_value.strip()
            if normalized_name not in _CANONICAL_HEADER_NAMES or not value:
                continue
            if len(value) > self._max_header_value_characters:
                raise ValueError("Music media header value is too long")
            accepted[normalized_name] = value
        result = tuple(
            (_CANONICAL_HEADER_NAMES[name], accepted[name])
            for name in _HEADER_ORDER
            if name in accepted
        )
        if sum(len(name) + len(value) for name, value in result) > self._max_total_characters:
            raise ValueError("Music media headers are too large")
        return result

    @staticmethod
    def _has_forbidden_characters(value: str) -> bool:
        return any(character in value for character in ("\r", "\n", "\0"))


DEFAULT_MUSIC_MEDIA_HEADER_POLICY = MusicMediaHeaderPolicy()
