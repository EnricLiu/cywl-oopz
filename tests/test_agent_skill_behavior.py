from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from cywl_oopz.features.agent.models import (
    AgentIdentity,
    AgentMessage,
    AgentModelRef,
    AgentRunLimits,
    AgentRunRequest,
    ModelCapability,
    ProviderProtocol,
)
from cywl_oopz.features.agent.prompts import AgentSystemPrompt
from cywl_oopz.features.agent.pydantic_ai_engine import PydanticAiAgentEngine
from cywl_oopz.features.agent.skills.library_tools import skill_library_tools
from cywl_oopz.features.agent.skills.models import (
    AgentSkill,
    AgentSkillBundle,
    AgentSkillDiscovery,
    AgentSkillInspection,
    AgentSkillInviteResult,
    AgentSkillLibrary,
    AgentSkillOwnedSummary,
    AgentSkillRevokeResult,
    AgentSkillShare,
    AgentSkillShareSummary,
    SkillAccessKind,
    SkillOwnershipKind,
    SkillShareStatus,
)
from cywl_oopz.features.agent.tools.models import (
    ToolCall,
    ToolDescriptor,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.chat.progress import ConversationProgressEvent, ProgressKind


class StaticRegistry:
    def __init__(self, model: FunctionModel) -> None:
        self._model = model

    async def model(self, reference: AgentModelRef) -> FunctionModel:
        del reference
        return self._model

    async def aclose(self) -> None:
        return None


class LibraryToolRuntime:
    def __init__(self, library: object) -> None:
        tools = skill_library_tools(library)  # type: ignore[arg-type]
        self._tools = {tool.descriptor.name: tool for tool in tools}
        self.calls: list[tuple[ToolCall, ToolExecutionContext]] = []

    def descriptors(self, names: tuple[str, ...]) -> tuple[ToolDescriptor, ...]:
        return tuple(self._tools[name].descriptor for name in sorted(names))

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        self.calls.append((call, context))
        tool = self._tools[call.name]
        arguments = tool.descriptor.input_model.model_validate(dict(call.arguments))
        try:
            output = await tool.execute(context, arguments)
        except ToolExecutionError as exc:
            return ToolExecutionResult(
                call.call_id,
                call.name,
                ToolExecutionStatus.FAILED,
                error_code=exc.error_code,
            )
        return ToolExecutionResult(
            call.call_id,
            call.name,
            ToolExecutionStatus.SUCCEEDED,
            output.model_dump(mode="json"),
        )


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[ConversationProgressEvent] = []

    async def emit(self, event: ConversationProgressEvent) -> None:
        self.events.append(event)


class ScriptedLibrary:
    def __init__(self) -> None:
        self.skill = AgentSkill(
            id=uuid4(),
            name="travel-planner",
            display_name="旅行规划",
            description="需要规划旅行时使用。",
            instructions="先确认目的地和日期，再按天规划。",
            version="1",
            revision=1,
            required_tools=frozenset(),
            resources=(),
            metadata={},
            ownership_kind=SkillOwnershipKind.PERSONAL,
            owner_person_id="owner",
        )
        now = datetime.now(UTC)
        self.share = AgentSkillShare(
            id=uuid4(),
            skill_id=self.skill.id,
            recipient_person_id="friend",
            status=SkillShareStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        self.operations: list[str] = []

    async def library(self, person_id: str) -> AgentSkillLibrary:
        self.operations.append(f"library:{person_id}")
        if person_id == "friend":
            return AgentSkillLibrary(
                owned=(),
                builtin=(),
                shared=(),
                pending_invitations=(self._share_summary(SkillShareStatus.PENDING),),
            )
        return AgentSkillLibrary(
            owned=(AgentSkillOwnedSummary(self._discovery(SkillAccessKind.OWNED), True),),
            builtin=(),
            shared=(),
        )

    async def create(self, person_id: str, **values: object) -> AgentSkill:
        assert person_id == "owner"
        self.operations.append("create")
        self.skill = replace(
            self.skill,
            name=str(values["name"]),
            display_name=str(values["display_name"]),
            description=str(values["description"]),
            instructions=str(values["instructions"]),
        )
        return self.skill

    async def inspect(
        self,
        person_id: str,
        skill_id: UUID,
        resource_key: str | None = None,
    ) -> AgentSkillInspection:
        assert person_id == "owner"
        assert skill_id == self.skill.id
        assert resource_key is None
        self.operations.append("inspect")
        return AgentSkillInspection(
            AgentSkillBundle(
                self._discovery(SkillAccessKind.OWNED),
                self.skill.instructions,
                (),
            ),
            active=True,
        )

    async def update(
        self,
        person_id: str,
        skill_id: UUID,
        expected_revision: int,
        **values: object,
    ) -> AgentSkill:
        assert (person_id, skill_id, expected_revision) == (
            "owner",
            self.skill.id,
            1,
        )
        self.operations.append("update")
        self.skill = replace(
            self.skill,
            instructions=str(values["instructions"]),
            revision=2,
        )
        return self.skill

    async def invite(
        self,
        person_id: str,
        skill_id: UUID,
        mentioned_person_ids: tuple[str, ...],
    ) -> AgentSkillInviteResult:
        assert (person_id, skill_id) == ("owner", self.skill.id)
        assert mentioned_person_ids == ("friend",)
        self.operations.append("invite")
        return AgentSkillInviteResult(
            self._discovery(SkillAccessKind.OWNED),
            (self.share,),
            notification_failures=1,
        )

    async def respond(
        self,
        person_id: str,
        share_id: UUID,
        *,
        accepted: bool,
    ) -> AgentSkillShareSummary:
        assert (person_id, share_id, accepted) == ("friend", self.share.id, True)
        self.operations.append("respond")
        self.share = replace(
            self.share,
            status=SkillShareStatus.ACCEPTED,
            responded_at=datetime.now(UTC),
        )
        return self._share_summary(SkillShareStatus.ACCEPTED)

    async def revoke(
        self,
        person_id: str,
        skill_id: UUID,
        mentioned_person_ids: tuple[str, ...],
        *,
        revoke_all: bool,
    ) -> AgentSkillRevokeResult:
        assert (person_id, skill_id) == ("owner", self.skill.id)
        assert mentioned_person_ids == ("friend",)
        assert revoke_all is False
        self.operations.append("revoke")
        return AgentSkillRevokeResult(
            self._discovery(SkillAccessKind.OWNED),
            (self.share,),
            notification_failures=1,
        )

    def _discovery(self, access: SkillAccessKind) -> AgentSkillDiscovery:
        return AgentSkillDiscovery(
            id=self.skill.id,
            name=self.skill.name,
            display_name=self.skill.display_name,
            description=self.skill.description,
            version=self.skill.version,
            revision=self.skill.revision,
            required_tools=self.skill.required_tools,
            access=access,
        )

    def _share_summary(self, status: SkillShareStatus) -> AgentSkillShareSummary:
        share = self.share
        if share.status is not status:
            share = replace(
                share,
                status=status,
                responded_at=(None if status is SkillShareStatus.PENDING else datetime.now(UTC)),
            )
        return AgentSkillShareSummary(
            share,
            self._discovery(SkillAccessKind.SHARED),
            active=True,
        )


def streaming_model(respond) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo):
        response = await respond(messages, info)
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {
                    index: DeltaToolCall(
                        name=part.tool_name,
                        json_args=json.dumps(part.args),
                        tool_call_id=part.tool_call_id,
                    )
                }

    return FunctionModel(stream_function=stream)


def tool_returns(messages: list[ModelMessage]) -> tuple[ToolReturnPart, ...]:
    return tuple(
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )


def request(
    *,
    person_id: str,
    prompt: str,
    enabled_tools: tuple[str, ...],
    mentioned_person_ids: tuple[str, ...] = (),
) -> AgentRunRequest:
    key = ConversationKey("channel", "area", "channel", person_id)
    return AgentRunRequest(
        run_id=uuid4(),
        thread_id=uuid4(),
        identity=AgentIdentity(
            person_id,
            key,
            source_message_id="source",
            transport_channel_id="channel",
            mentioned_person_ids=mentioned_person_ids,
        ),
        model=AgentModelRef(
            provider_id=uuid4(),
            model_id=uuid4(),
            provider_alias="function",
            model_alias="behavior",
            remote_model_name="behavior",
            protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
            capabilities=frozenset({ModelCapability.TOOL_CALLING}),
            fallback_model_id=None,
        ),
        prompt=prompt,
        context=(
            AgentMessage(
                "system",
                "text",
                {"text": AgentSystemPrompt("你是初音未来。").render()},
            ),
        ),
        enabled_tools=tuple(sorted(enabled_tools)),
        limits=AgentRunLimits(
            timeout_seconds=5,
            max_model_requests=8,
            max_tool_calls=8,
        ),
    )


@pytest.mark.asyncio
async def test_function_model_completes_create_inspect_update_chain() -> None:
    library = ScriptedLibrary()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert "必须先调用 `inspect_agent_skill`" in (info.instructions or "")
        returns = tool_returns(messages)
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "create_agent_skill",
                        {
                            "name": "travel-planner",
                            "display_name": "旅行规划",
                            "description": "需要规划旅行时使用。",
                            "instructions": "先确认目的地和日期，再规划。",
                            "required_tools": [],
                        },
                        "create",
                    )
                ]
            )
        if returns[-1].tool_name == "create_agent_skill":
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "inspect_agent_skill",
                        {"skill_id": str(library.skill.id)},
                        "inspect",
                    )
                ]
            )
        if returns[-1].tool_name == "inspect_agent_skill":
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "update_agent_skill",
                        {
                            "skill_id": str(library.skill.id),
                            "expected_revision": 1,
                            "instructions": "先确认目的地、日期和预算，再按天规划。",
                        },
                        "update",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("技能已创建并更新，将从下一轮生效。")])

    runtime = LibraryToolRuntime(library)
    engine = PydanticAiAgentEngine(StaticRegistry(streaming_model(respond)), runtime)
    progress = RecordingProgress()

    result = await engine.run(
        request(
            person_id="owner",
            prompt="创建旅行规划技能，再把预算步骤补进去",
            enabled_tools=(
                "create_agent_skill",
                "inspect_agent_skill",
                "update_agent_skill",
            ),
        ),
        progress,
    )

    assert result.output == "技能已创建并更新，将从下一轮生效。"
    assert library.operations == ["create", "inspect", "update"]
    assert result.tool_calls == 3
    assert result.intermediate_messages == ()
    assert [event.kind for event in progress.events].count(ProgressKind.TOOL_SUCCEEDED) == 3


@pytest.mark.asyncio
async def test_function_model_completes_invite_accept_and_revoke_chains() -> None:
    library = ScriptedLibrary()
    runtime = LibraryToolRuntime(library)

    def model_for(action: str) -> FunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert "真实 `@` 提及" in (info.instructions or "")
            returns = tool_returns(messages)
            if not returns:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "list_agent_skill_library",
                            {},
                            f"list-{action}",
                        )
                    ]
                )
            if returns[-1].tool_name == "list_agent_skill_library":
                arguments: dict[str, object]
                if action == "invite":
                    tool_name = "invite_agent_skill_share"
                    arguments = {"skill_id": str(library.skill.id)}
                elif action == "accept":
                    tool_name = "respond_agent_skill_share"
                    arguments = {
                        "share_id": str(library.share.id),
                        "decision": "accept",
                    }
                else:
                    tool_name = "revoke_agent_skill_share"
                    arguments = {"skill_id": str(library.skill.id)}
                return ModelResponse(parts=[ToolCallPart(tool_name, arguments, action)])
            return ModelResponse(parts=[TextPart(f"{action} 完成")])

        return streaming_model(respond)

    scenarios = (
        (
            "invite",
            "owner",
            "把旅行规划技能分享给 @朋友",
            ("friend",),
            "invite_agent_skill_share",
        ),
        (
            "accept",
            "friend",
            "接受旅行规划技能邀请",
            (),
            "respond_agent_skill_share",
        ),
        (
            "revoke",
            "owner",
            "撤销刚才的旅行规划技能分享",
            ("friend",),
            "revoke_agent_skill_share",
        ),
    )
    for action, person_id, prompt, mentions, mutation_tool in scenarios:
        engine = PydanticAiAgentEngine(StaticRegistry(model_for(action)), runtime)
        result = await engine.run(
            request(
                person_id=person_id,
                prompt=prompt,
                enabled_tools=("list_agent_skill_library", mutation_tool),
                mentioned_person_ids=mentions,
            )
        )
        assert result.output == f"{action} 完成"
        assert result.tool_calls == 2

    assert library.operations == [
        "library:owner",
        "invite",
        "library:friend",
        "respond",
        "library:owner",
        "revoke",
    ]
    mutation_contexts = [
        context for call, context in runtime.calls if call.name != "list_agent_skill_library"
    ]
    assert mutation_contexts[0].identity.mentioned_person_ids == ("friend",)
    assert mutation_contexts[1].identity.person_id == "friend"
