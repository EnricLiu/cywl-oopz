"""Route pinned voice configuration to one protocol-specific Provider adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cywl_oopz.features.voice.errors import VoiceProviderConfigurationError
from cywl_oopz.features.voice.ports import RealtimeVoiceProvider, VoiceSessionRuntimeContext
from cywl_oopz.features.voice.prompt import VoicePromptCompiler
from cywl_oopz.features.voice.settings import VoiceProviderProtocol

from .qwen_audio import QwenAudioProviderBuilder
from .qwen_omni import QwenOmniProviderBuilder


class ConfiguredVoiceProviderBuilder:
    """Keep protocol selection outside the Provider-neutral session coordinator."""

    def __init__(
        self,
        *,
        tool_schemas: Sequence[Mapping[str, object]] = (),
        prompt_compiler: VoicePromptCompiler | None = None,
    ) -> None:
        prompts = prompt_compiler or VoicePromptCompiler()
        self._builders = {
            VoiceProviderProtocol.QWEN_OMNI_REALTIME_WS: QwenOmniProviderBuilder(
                prompts,
                tool_schemas=tool_schemas,
            ),
            VoiceProviderProtocol.QWEN_AUDIO_REALTIME_WS: QwenAudioProviderBuilder(
                prompts,
                tool_schemas=tool_schemas,
            ),
        }

    def __call__(self, context: VoiceSessionRuntimeContext) -> RealtimeVoiceProvider:
        protocol = context.configuration.provider.protocol
        builder = self._builders.get(protocol)
        if builder is None:
            raise VoiceProviderConfigurationError(
                f"Voice Provider protocol is not implemented: {protocol.value}"
            )
        return builder(context)
