from __future__ import annotations

import asyncio
import os
from types import MappingProxyType
from uuid import uuid4

import pytest
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.voice.errors import VoiceProviderConfigurationError
from cywl_oopz.features.voice.events import VoiceSessionFinished, VoiceSessionReady
from cywl_oopz.features.voice.models import (
    VoiceChannelKey,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.features.voice.settings import (
    VoiceChannelConfiguration,
    VoiceDuplexMode,
    VoiceModelConfiguration,
    VoiceProviderConfiguration,
    VoiceProviderProtocol,
    VoiceStartConfiguration,
)
from cywl_oopz.integrations.voice.qwen_omni import QwenOmniRealtimeProvider
from cywl_oopz.integrations.voice.qwen_protocol import QwenOmniConfig
from cywl_oopz.storage.models import VoiceModelRecord, VoiceProviderRecord
from cywl_oopz.storage.url import normalize_asyncpg_url


async def _load_live_config() -> QwenOmniConfig:
    env_names = ("CYWL_QWEN_ENDPOINT", "CYWL_QWEN_API_KEY", "CYWL_QWEN_MODEL")
    if all(os.getenv(name) for name in env_names):
        return QwenOmniConfig(
            endpoint=os.environ["CYWL_QWEN_ENDPOINT"],
            api_key=os.environ["CYWL_QWEN_API_KEY"],
            model=os.environ["CYWL_QWEN_MODEL"],
            voice=os.getenv("CYWL_QWEN_VOICE", "Cherry"),
        )

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        pytest.skip("Qwen live smoke requires complete CYWL_QWEN_* values or DATABASE_URL")
    engine = create_async_engine(normalize_asyncpg_url(database_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            row = (
                await session.execute(
                    select(VoiceModelRecord, VoiceProviderRecord)
                    .join(
                        VoiceProviderRecord,
                        VoiceProviderRecord.id == VoiceModelRecord.provider_id,
                    )
                    .where(
                        VoiceProviderRecord.protocol == VoiceProviderProtocol.QWEN_OMNI_REALTIME_WS,
                        VoiceProviderRecord.enabled.is_(True),
                        VoiceModelRecord.enabled.is_(True),
                        VoiceModelRecord.is_application_default.is_(True),
                    )
                )
            ).one_or_none()
            if row is None:
                pytest.skip("database has no enabled application-default Qwen Omni voice model")
            model, provider = row
            configuration = VoiceStartConfiguration(
                provider=VoiceProviderConfiguration(
                    provider.id,
                    provider.alias,
                    provider.display_name,
                    provider.protocol,
                    provider.endpoint,
                    MappingProxyType(dict(provider.credentials)),
                    MappingProxyType(dict(provider.config)),
                ),
                model=VoiceModelConfiguration(
                    model.id,
                    model.provider_id,
                    model.alias,
                    model.remote_model_name,
                    model.display_name,
                    model.mode,
                    MappingProxyType(dict(model.capabilities)),
                    MappingProxyType(dict(model.audio_config)),
                    MappingProxyType(dict(model.prompt_config)),
                    MappingProxyType(dict(model.limits)),
                ),
                channel=VoiceChannelConfiguration(
                    VoiceChannelKey("live-test-area", "live-test-voice"),
                    "voice_readonly_v1",
                    300,
                ),
                voice_id=os.getenv("CYWL_QWEN_VOICE", ""),
                duplex_mode=VoiceDuplexMode.FULL,
                delegated_agent_model_id=None,
            )
            try:
                return QwenOmniConfig.from_start_configuration(configuration)
            except VoiceProviderConfigurationError as exc:
                pytest.skip(f"database Qwen Omni configuration is incomplete: {type(exc).__name__}")
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_qwen_live_handshake_and_clean_finish() -> None:
    if os.getenv("CYWL_RUN_LIVE_QWEN_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_QWEN_TESTS=1 to run the Qwen realtime smoke")
    load_dotenv(find_dotenv(usecwd=True), override=False)
    config = await _load_live_config()
    provider = QwenOmniRealtimeProvider(
        config,
        "你正在执行连接测试。不要主动说话。",
    )
    descriptor = VoiceSessionDescriptor(
        uuid4(),
        "live-test-person",
        VoiceChannelKey("live-test-area", "live-test-voice"),
        VoiceTextAddress("live-test-area", "live-test-text"),
    )
    session = await provider.connect(descriptor)
    events = session.events()
    try:
        async with asyncio.timeout(15):
            ready = await anext(events)
            assert isinstance(ready, VoiceSessionReady)
            await session.finish()
            terminal = await anext(events)
            assert isinstance(terminal, VoiceSessionFinished)
    finally:
        await provider.aclose()
