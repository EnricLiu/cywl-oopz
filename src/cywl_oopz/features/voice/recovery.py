"""Provider retry and replacement-media ownership for realtime sessions."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from contextlib import suppress

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.settings import VoiceSettings

from .errors import (
    VoiceProviderAuthenticationError,
    VoiceProviderConfigurationError,
    VoiceProviderDisconnectedError,
)
from .models import VoiceMediaEndReason
from .ports import (
    RealtimeVoiceProvider,
    RealtimeVoiceSession,
    VoiceMediaGateway,
    VoiceMediaSession,
    VoiceSessionRuntimeContext,
)
from .runtime_events import _MediaRecovered, _MediaRecoveryFailed

logger = logging.getLogger(__name__)

ProviderBuilder = Callable[[VoiceSessionRuntimeContext], RealtimeVoiceProvider]
MediaRecoveryEvent = _MediaRecovered | _MediaRecoveryFailed


class VoiceProviderConnector:
    """Retry connection failures while closing every unsuccessful provider."""

    def __init__(self, settings: VoiceSettings) -> None:
        self._settings = settings

    async def connect(
        self,
        context: VoiceSessionRuntimeContext,
        builder: ProviderBuilder,
        on_attempt: Callable[[], None],
    ) -> tuple[RealtimeVoiceProvider, RealtimeVoiceSession]:
        last_error: Exception | None = None
        for attempt in range(1, self._settings.provider_connect_attempts + 1):
            on_attempt()
            provider = builder(context)
            try:
                session = await provider.connect(context.descriptor)
            except asyncio.CancelledError:
                with suppress(Exception):
                    await provider.aclose()
                raise
            except Exception as exc:
                last_error = exc
                with suppress(Exception):
                    await provider.aclose()
                logger.warning(
                    "Voice Provider connect failed: session=%s attempt=%d error=%s",
                    opaque_ref(str(context.descriptor.session_id)),
                    attempt,
                    exception_kind(exc),
                    exc_info=True,
                )
                if (
                    isinstance(
                        exc,
                        (VoiceProviderAuthenticationError, VoiceProviderConfigurationError),
                    )
                    or attempt >= self._settings.provider_connect_attempts
                ):
                    break
                delay = min(1.5, 0.2 * (2 ** (attempt - 1)))
                delay *= random.uniform(0.8, 1.2)
                logger.warning(
                    "Voice Provider connect retry: session=%s attempt=%d error=%s",
                    opaque_ref(str(context.descriptor.session_id)),
                    attempt,
                    exception_kind(exc),
                )
                await asyncio.sleep(delay)
            else:
                return provider, session
        raise VoiceProviderDisconnectedError(
            "Voice Provider connection attempts exhausted"
        ) from last_error


class VoiceMediaRecovery:
    """Own replacement transports until the control loop accepts or closes them."""

    def __init__(
        self,
        context: VoiceSessionRuntimeContext,
        settings: VoiceSettings,
        media_gateway: VoiceMediaGateway,
        publish: Callable[[MediaRecoveryEvent], Awaitable[None]],
    ) -> None:
        self._context = context
        self._settings = settings
        self._media_gateway = media_gateway
        self._publish = publish
        self.pending: VoiceMediaSession | None = None

    async def run(
        self,
        generation: int,
        reason: VoiceMediaEndReason,
        old_media: VoiceMediaSession,
    ) -> None:
        replacement: VoiceMediaSession | None = None
        try:
            async with asyncio.timeout(self._settings.owner_leave_grace_seconds):
                await old_media.aclose()
                replacement = await self._media_gateway.open(
                    self._context.descriptor,
                    self._context.lease,
                )
            self.pending = replacement
            await self._publish(_MediaRecovered(generation, replacement))
        except asyncio.CancelledError:
            if replacement is not None:
                await self._discard(replacement)
                if self.pending is replacement:
                    self.pending = None
            raise
        except TimeoutError:
            if replacement is not None:
                await self._discard(replacement)
            await self._publish(_MediaRecoveryFailed(generation, reason, "timeout"))
        except Exception as exc:
            if replacement is not None:
                await self._discard(replacement)
            logger.warning(
                "Voice media recovery failed: session=%s generation=%d reason=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                generation,
                reason.value,
                exception_kind(exc),
                exc_info=True,
            )
            await self._publish(_MediaRecoveryFailed(generation, reason, exception_kind(exc)))

    async def _discard(self, media: VoiceMediaSession) -> None:
        try:
            async with asyncio.timeout(min(0.25, self._settings.stop_timeout_seconds / 4)):
                await media.aclose()
        except TimeoutError:
            logger.warning(
                "Replacement voice media close exceeded cleanup budget: session=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
            )
        except Exception as exc:
            logger.warning(
                "Replacement voice media close failed: session=%s error=%s",
                opaque_ref(str(self._context.descriptor.session_id)),
                exception_kind(exc),
                exc_info=True,
            )
