"""OOPZ area discovery adapter for administration use cases."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from oopz_sdk import OopzBot

from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.features.admin.initialization import ChannelCatalogError
from cywl_oopz.features.admin.models import AreaChannelCatalog, ChannelKey

logger = logging.getLogger(__name__)


class OopzAreaChannelCatalog:
    """Project SDK channel groups into a small immutable catalog."""

    _TEXT_TYPES = frozenset({"TEXT"})
    _VOICE_TYPES = frozenset({"VOICE", "AUDIO"})
    _SUPPORTED_TYPES = _TEXT_TYPES | _VOICE_TYPES

    def __init__(self, bot: OopzBot, *, timeout_seconds: float = 10.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Catalog timeout must be positive")
        self._bot = bot
        self._timeout_seconds = timeout_seconds

    async def discover(self, area_id: str) -> AreaChannelCatalog:
        normalized_area = area_id.strip()
        try:
            async with asyncio.timeout(self._timeout_seconds):
                groups = await self._bot.areas.get_area_channels(normalized_area)
        except TimeoutError as exc:
            logger.warning("OOPZ area channel discovery timed out: area=%s", opaque_ref(area_id))
            raise ChannelCatalogError("OOPZ area channel discovery timed out") from exc
        except Exception as exc:
            logger.warning(
                "OOPZ area channel discovery failed: area=%s error=%s",
                opaque_ref(area_id),
                type(exc).__name__,
            )
            raise ChannelCatalogError("OOPZ area channel discovery failed") from exc

        classified: dict[str, str] = {}
        for group in groups:
            for channel in getattr(group, "channels", ()) or ():
                channel_id = str(getattr(channel, "channel_id", "")).strip()
                channel_type = self._channel_type(getattr(channel, "channel_type", ""))
                if not channel_id or channel_type not in self._SUPPORTED_TYPES:
                    continue
                previous = classified.setdefault(channel_id, channel_type)
                if previous != channel_type:
                    logger.warning(
                        "OOPZ channel appeared with conflicting types: area=%s channel=%s",
                        opaque_ref(area_id),
                        opaque_ref(area_id, channel_id),
                    )

        try:
            text_channels = tuple(
                ChannelKey(normalized_area, channel_id)
                for channel_id, channel_type in classified.items()
                if channel_type in self._TEXT_TYPES
            )
            voice_channels = tuple(
                ChannelKey(normalized_area, channel_id)
                for channel_id, channel_type in classified.items()
                if channel_type in self._VOICE_TYPES
            )
            catalog = AreaChannelCatalog(normalized_area, text_channels, voice_channels)
        except ValueError as exc:
            logger.warning(
                "OOPZ returned an invalid area channel identifier: area=%s error=%s",
                opaque_ref(area_id),
                type(exc).__name__,
            )
            raise ChannelCatalogError("OOPZ returned an invalid channel catalog") from exc
        logger.debug(
            "OOPZ area channels discovered: area=%s text=%s voice=%s",
            opaque_ref(area_id),
            len(catalog.text_channels),
            len(catalog.voice_channels),
        )
        return catalog

    @staticmethod
    def _channel_type(value: Any) -> str:
        raw = getattr(value, "value", value)
        return str(raw).strip().upper()
