import pytest

from cywl_oopz.features.agent.prompts import AgentSystemPrompt


def test_system_prompt_preserves_persona_and_defines_agent_loop_contract() -> None:
    rendered = AgentSystemPrompt("你是一个喜欢冷笑话的社区 DJ。").render()

    assert rendered.startswith("你是一个喜欢冷笑话的社区 DJ。")
    assert "## Agent 工作循环" in rendered
    assert "判断是否需要工具" in rendered
    assert "`ok=true`" in rendered
    assert "`ok=false`" in rendered
    assert "不要无变化地重复同一调用" in rendered
    assert "结束循环并给出最终回复" in rendered
    assert "绝不伪造调用、结果或成功状态" in rendered
    assert "不要输出隐藏的逐步思考" in rendered


def test_system_prompt_rejects_an_empty_persona() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        AgentSystemPrompt("  ")
