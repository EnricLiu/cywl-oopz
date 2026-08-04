from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from dotenv import find_dotenv, load_dotenv

from cywl_oopz.features.voice.events import VoiceSessionFinished, VoiceSessionReady
from cywl_oopz.features.voice.models import (
    VoiceChannelKey,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.integrations.voice.qwen_omni import QwenOmniRealtimeProvider
from cywl_oopz.integrations.voice.qwen_protocol import QwenOmniConfig


@pytest.mark.asyncio
async def test_qwen_live_handshake_and_clean_finish() -> None:
    if os.getenv("CYWL_RUN_LIVE_QWEN_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_QWEN_TESTS=1 to run the Qwen realtime smoke")
    load_dotenv(find_dotenv(usecwd=True), override=False)
    required = ("CYWL_QWEN_ENDPOINT", "CYWL_QWEN_API_KEY", "CYWL_QWEN_MODEL")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip("live Qwen smoke configuration is incomplete")

    config = QwenOmniConfig(
        endpoint=os.environ["CYWL_QWEN_ENDPOINT"],
        api_key=os.environ["CYWL_QWEN_API_KEY"],
        model=os.environ["CYWL_QWEN_MODEL"],
        voice=os.getenv("CYWL_QWEN_VOICE", "Cherry"),
    )
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
