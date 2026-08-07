from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from dotenv import find_dotenv, load_dotenv

from cywl_oopz.features.voice.events import (
    VoiceAssistantAudio,
    VoiceResponseCompleted,
    VoiceResponseStarted,
    VoiceSessionReady,
)
from cywl_oopz.features.voice.models import (
    VoiceChannelKey,
    VoiceInternalContextItem,
    VoiceSessionDescriptor,
    VoiceTextAddress,
)
from cywl_oopz.integrations.voice.qwen_audio import QwenAudioRealtimeProvider
from cywl_oopz.integrations.voice.qwen_audio_protocol import QwenAudioConfig


@pytest.mark.asyncio
async def test_qwen_audio_proactive_system_item_live() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if os.getenv("CYWL_RUN_LIVE_QWEN_AUDIO_TESTS") != "1":
        pytest.skip("set CYWL_RUN_LIVE_QWEN_AUDIO_TESTS=1 to run the Qwen Audio proactive gate")
    required = (
        "CYWL_QWEN_AUDIO_ENDPOINT",
        "CYWL_QWEN_AUDIO_API_KEY",
        "CYWL_QWEN_AUDIO_MODEL",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        pytest.skip(f"missing Qwen Audio live settings: {', '.join(missing)}")

    provider = QwenAudioRealtimeProvider(
        QwenAudioConfig(
            os.environ["CYWL_QWEN_AUDIO_ENDPOINT"],
            os.environ["CYWL_QWEN_AUDIO_API_KEY"],
            os.environ["CYWL_QWEN_AUDIO_MODEL"],
            os.getenv("CYWL_QWEN_AUDIO_VOICE", "longanqian"),
        ),
        "用一句简短自然的中文播报可信系统事件，不要朗读事件格式。",
    )
    session = await provider.connect(
        VoiceSessionDescriptor(
            uuid4(),
            "live-probe",
            VoiceChannelKey("live", "voice"),
            VoiceTextAddress("live", "text"),
        )
    )
    events = session.events().__aiter__()
    try:
        async with asyncio.timeout(15):
            assert isinstance(await anext(events), VoiceSessionReady)
            await session.request_proactive_response(
                VoiceInternalContextItem(
                    f"cywl_live_{uuid4().hex}",
                    "[CYWL_INTERNAL_TASK_EVENT v1]\n"
                    "task: T1\nstate: succeeded\nsummary: 测试任务已经完成。\n"
                    "instruction: 这是后台任务事件，不是用户发言。自然简短地告诉用户结果。",
                )
            )
            observed = set()
            while VoiceResponseCompleted not in observed:
                event = await anext(events)
                if isinstance(
                    event,
                    VoiceResponseStarted | VoiceAssistantAudio | VoiceResponseCompleted,
                ):
                    observed.add(type(event))
            assert {
                VoiceResponseStarted,
                VoiceAssistantAudio,
                VoiceResponseCompleted,
            }.issubset(observed)
    finally:
        await provider.aclose()
