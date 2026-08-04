from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from cywl_oopz.features.voice.models import VoiceChannelKey
from cywl_oopz.features.voice.prompt import CYWL_VOICE_SYSTEM_PROMPT, VoicePromptCompiler
from cywl_oopz.integrations.voice.fake import FakeVoiceConfigurationRepository


@pytest.mark.asyncio
async def test_voice_prompt_is_spoken_bounded_and_explains_task_delegation() -> None:
    configuration = await FakeVoiceConfigurationRepository().resolve_start_configuration(
        "person",
        VoiceChannelKey("area", "voice"),
    )

    prompt = VoicePromptCompiler().compile(configuration)

    assert prompt == CYWL_VOICE_SYSTEM_PROMPT
    assert "初音未来" in prompt
    assert "一到三句" in prompt
    assert "delegate_agent_task" in prompt
    assert "只代表已经排队，不代表" in prompt
    assert "不要编造任务状态" in prompt
    assert "不要朗读 Markdown" in prompt


@pytest.mark.asyncio
async def test_voice_prompt_appends_only_bounded_trusted_model_instructions() -> None:
    configuration = await FakeVoiceConfigurationRepository().resolve_start_configuration(
        "person", VoiceChannelKey("area", "voice")
    )
    model = replace(
        configuration.model,
        prompt_config=MappingProxyType({"additional_instructions": "称呼用户为制作人。"}),
    )

    prompt = VoicePromptCompiler().compile(replace(configuration, model=model))

    assert prompt.endswith("称呼用户为制作人。")

    oversized = replace(
        configuration,
        model=replace(
            configuration.model,
            prompt_config=MappingProxyType({"additional_instructions": "x" * 4_001}),
        ),
    )
    with pytest.raises(ValueError):
        VoicePromptCompiler().compile(oversized)
