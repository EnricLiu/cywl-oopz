"""Qwen-Audio Realtime adapter with documented proactive context injection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from websockets.asyncio.client import connect

from cywl_oopz.features.voice.ports import VoiceSessionRuntimeContext
from cywl_oopz.features.voice.prompt import VoicePromptCompiler

from .qwen_audio_protocol import QwenAudioConfig
from .qwen_omni import QwenConnector, QwenOmniRealtimeProvider


class QwenAudioRealtimeProvider(QwenOmniRealtimeProvider):
    """Use shared Qwen event pumps with Qwen-Audio's documented system-item surface."""

    def __init__(
        self,
        config: QwenAudioConfig,
        instructions: str,
        *,
        tool_schemas: Sequence[Mapping[str, object]] = (),
        connector: QwenConnector = connect,
    ) -> None:
        super().__init__(
            config,
            instructions,
            tool_schemas=tool_schemas,
            connector=connector,
            proactive_context=True,
            send_finish_event=False,
        )


class QwenAudioProviderBuilder:
    def __init__(
        self,
        prompt_compiler: VoicePromptCompiler | None = None,
        *,
        tool_schemas: Sequence[Mapping[str, object]] = (),
        connector: QwenConnector | None = None,
    ) -> None:
        self._prompts = prompt_compiler or VoicePromptCompiler()
        self._tool_schemas = tuple(tool_schemas)
        self._connector = connector

    def __call__(self, context: VoiceSessionRuntimeContext) -> QwenAudioRealtimeProvider:
        configuration = context.configuration
        kwargs = {"connector": self._connector} if self._connector is not None else {}
        return QwenAudioRealtimeProvider(
            QwenAudioConfig.from_start_configuration(configuration),
            self._prompts.compile(configuration),
            tool_schemas=self._tool_schemas,
            **kwargs,
        )
