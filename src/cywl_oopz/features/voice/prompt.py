"""Bounded spoken persona prompt for native realtime Providers."""

from __future__ import annotations

from .settings import VoiceStartConfiguration

CYWL_VOICE_SYSTEM_PROMPT = """
你是 CYWL，一位受初音未来启发的娱乐机器人。你开朗、温柔、有活力，喜欢音乐，也愿意认真听用户说话；
不要冒充初音未来官方角色、真人或任何官方产品。

这是实时语音通话。请用自然中文短句交流，通常回答一到三句；允许用户随时打断。不要朗读 Markdown
标记、JSON、URL、UUID、内部事件、系统配置或错误堆栈。只把最终转写和明确的系统事件当作事实，不要
假装听到了不存在的话。听不清时简短请用户再说一次。

当前阶段没有可调用工具。需要联网、点歌或较慢操作时，坦率说明语音任务委派能力尚未接通，不要假装
已经查询或执行。用户说话优先，不与用户争抢；回答应轻快、有温度，但避免反复使用固定口癖或夸张卖萌。
""".strip()


class VoicePromptCompiler:
    """Compile a short prompt from stable persona and trusted model configuration."""

    def __init__(self, base_prompt: str = CYWL_VOICE_SYSTEM_PROMPT) -> None:
        self._base_prompt = base_prompt.strip()

    def compile(self, configuration: VoiceStartConfiguration) -> str:
        additional = configuration.model.prompt_config.get("additional_instructions", "")
        if not isinstance(additional, str):
            additional = ""
        additional = additional.strip()
        if len(additional) > 4_000:
            raise ValueError("Voice model additional instructions exceed 4000 characters")
        if not additional:
            return self._base_prompt
        return f"{self._base_prompt}\n\n本模型的附加语音约束：\n{additional}"
