from __future__ import annotations

import pytest
from oopz_sdk import VoiceCapabilities

from cywl_oopz.core.errors import ConfigurationError
from cywl_oopz.integrations.oopz.voice_capabilities import OopzVoiceCapabilityGate


def capabilities(**overrides: bool | int) -> VoiceCapabilities:
    values: dict[str, bool | int] = {
        "feature_version": 1,
        "remote_audio_subscription": True,
        "person_audio_subscription": True,
        "streaming_pcm_output": True,
        "playback_cursor": True,
        "typed_playback_handle": True,
    }
    values.update(overrides)
    return VoiceCapabilities(**values)


def test_voice_capability_gate_accepts_sdk_feature_version_one() -> None:
    OopzVoiceCapabilityGate().validate(capabilities())


def test_voice_capability_gate_reports_version_and_missing_primitives() -> None:
    with pytest.raises(ConfigurationError) as error:
        OopzVoiceCapabilityGate().validate(
            capabilities(
                feature_version=0,
                person_audio_subscription=False,
                streaming_pcm_output=False,
            )
        )

    message = str(error.value)
    assert "feature_version>=1" in message
    assert "person_audio_subscription" in message
    assert "streaming_pcm_output" in message
