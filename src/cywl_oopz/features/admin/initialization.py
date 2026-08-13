"""Idempotent channel initialization use cases."""

from __future__ import annotations

import logging

from cywl_oopz.core.errors import CywlError
from cywl_oopz.core.observability import opaque_ref

from .models import (
    AreaInitializationResult,
    ChannelInitializationResult,
    ChannelKey,
)
from .ports import AreaChannelCatalogPort, ChannelInitializationRepository

logger = logging.getLogger(__name__)


class ChannelCatalogError(CywlError):
    """Raised when OOPZ cannot provide a usable area channel catalog."""


class ChannelInitializationService:
    """Coordinate OOPZ discovery before short PostgreSQL writes."""

    def __init__(
        self,
        catalog: AreaChannelCatalogPort,
        repository: ChannelInitializationRepository,
    ) -> None:
        self._catalog = catalog
        self._repository = repository

    async def initialize_channel(
        self,
        channel: ChannelKey,
    ) -> ChannelInitializationResult:
        result = await self._repository.initialize_text_channel(channel)
        logger.info(
            "Text channel initialization completed: channel=%s created=%s",
            opaque_ref(channel.area_id, channel.channel_id),
            result.created,
        )
        return result

    async def initialize_area(self, area_id: str) -> AreaInitializationResult:
        catalog = await self._catalog.discover(area_id)
        result = await self._repository.initialize_area(catalog)
        logger.info(
            "Area channel initialization completed: area=%s text_created=%s "
            "text_existing=%s voice_created=%s voice_existing=%s",
            opaque_ref(area_id),
            result.text_created,
            result.text_existing,
            result.voice_created,
            result.voice_existing,
        )
        return result
