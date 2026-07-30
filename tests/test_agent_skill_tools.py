from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from cywl_oopz.features.agent.models import (
    AgentIdentity,
    AgentRunLimits,
)
from cywl_oopz.features.agent.skills.errors import AgentSkillRevisionConflictError
from cywl_oopz.features.agent.skills.library_tools import (
    CreateAgentSkillTool,
    InspectAgentSkillInput,
    InspectAgentSkillTool,
    InviteAgentSkillShareInput,
    InviteAgentSkillShareTool,
)
from cywl_oopz.features.agent.skills.models import (
    AgentSkill,
    AgentSkillBundle,
    AgentSkillDiscovery,
    AgentSkillInspection,
    AgentSkillInviteResult,
    AgentSkillResource,
    AgentSkillResourceManifest,
    AgentSkillShare,
    SkillAccessKind,
    SkillOwnershipKind,
    SkillResourceKind,
    SkillShareStatus,
)
from cywl_oopz.features.agent.skills.scope import (
    AgentSkillRunScope,
    AgentSkillScopeError,
)
from cywl_oopz.features.agent.skills.tools import (
    LoadAgentSkillTool,
    ReadAgentSkillResourceTool,
)
from cywl_oopz.features.agent.tools.executor import ToolExecutor
from cywl_oopz.features.agent.tools.models import (
    ToolCall,
    ToolExecution,
    ToolExecutionClaim,
    ToolExecutionContext,
    ToolExecutionStatus,
)
from cywl_oopz.features.agent.tools.policy import ToolPolicy
from cywl_oopz.features.agent.tools.registry import ToolRegistry
from cywl_oopz.features.chat.models import ConversationKey


class InMemoryExecutionRepository:
    def __init__(self) -> None:
        self.records: dict[tuple[UUID, str], ToolExecution] = {}

    async def claim(self, execution: ToolExecution) -> ToolExecutionClaim:
        key = (execution.run_id, execution.call_id)
        existing = self.records.get(key)
        if existing is not None:
            return ToolExecutionClaim(existing, False)
        self.records[key] = execution
        return ToolExecutionClaim(execution, True)

    async def finish(
        self,
        run_id: UUID,
        call_id: str,
        status: ToolExecutionStatus,
        *,
        output: dict[str, object] | None,
        error_code: str,
    ) -> ToolExecution:
        key = (run_id, call_id)
        execution = replace(
            self.records[key],
            status=status,
            output_payload=output,
            error_code=error_code,
            finished_at=datetime.now(UTC),
        )
        self.records[key] = execution
        return execution


class InMemorySkillReadRepository:
    def __init__(self, skills: tuple[AgentSkill, ...]) -> None:
        self.skills = {skill.id: skill for skill in skills}
        self.bundle_reads = 0
        self.resource_reads = 0

    async def list_accessible(self, person_id: str) -> tuple[AgentSkillDiscovery, ...]:
        del person_id
        return tuple(_discovery(skill) for skill in self.skills.values())

    async def load_accessible_bundle(
        self,
        person_id: str,
        skill_id: UUID,
        revision: int,
    ) -> AgentSkillBundle | None:
        del person_id
        self.bundle_reads += 1
        skill = self.skills.get(skill_id)
        if skill is None:
            return None
        if skill.revision != revision:
            raise AgentSkillRevisionConflictError("changed")
        return AgentSkillBundle(
            discovery=_discovery(skill),
            instructions=skill.instructions,
            resources=tuple(
                AgentSkillResourceManifest(
                    id=resource.id,
                    key=resource.key,
                    display_name=resource.display_name,
                    description=resource.description,
                    kind=resource.kind,
                    media_type=resource.media_type,
                    position=resource.position,
                )
                for resource in skill.resources
            ),
        )

    async def read_accessible_resource(
        self,
        person_id: str,
        skill_id: UUID,
        resource_id: UUID,
        revision: int,
    ) -> AgentSkillResource | None:
        del person_id
        self.resource_reads += 1
        skill = self.skills.get(skill_id)
        if skill is None:
            return None
        if skill.revision != revision:
            raise AgentSkillRevisionConflictError("changed")
        return next(
            (resource for resource in skill.resources if resource.id == resource_id),
            None,
        )


def make_skill(
    name: str = "web-research",
    *,
    instructions: str = "Search, read, and cite.",
    resource_content: str = "Prefer primary sources.",
) -> AgentSkill:
    resource = AgentSkillResource(
        id=uuid4(),
        key="source-guide",
        display_name="来源指南",
        description="需要判断来源质量时读取。",
        kind=SkillResourceKind.REFERENCE,
        media_type="text/markdown",
        content=resource_content,
        position=1,
    )
    return AgentSkill(
        id=uuid4(),
        name=name,
        display_name="网页研究" if name == "web-research" else "音乐策划",
        description="Complete one domain workflow.",
        instructions=instructions,
        version="1",
        revision=1,
        required_tools=frozenset(),
        resources=(resource,),
        metadata={},
    )


def make_scope(
    skills: tuple[AgentSkill, ...],
    *,
    available: tuple[AgentSkill, ...] | None = None,
    repository: InMemorySkillReadRepository | None = None,
    max_activations: int = 3,
    max_resources: int = 4,
    max_instruction_characters: int = 12_000,
    max_resource_characters: int = 12_000,
    max_context_characters: int = 24_000,
) -> AgentSkillRunScope:
    repository = repository or InMemorySkillReadRepository(skills)
    return AgentSkillRunScope(
        repository,
        "person",
        tuple(_discovery(skill) for skill in (skills if available is None else available)),
        max_activations=max_activations,
        max_resources=max_resources,
        max_instruction_characters=max_instruction_characters,
        max_resource_characters=max_resource_characters,
        max_context_characters=max_context_characters,
    )


def execution_context(
    scope: AgentSkillRunScope | None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        run_id=uuid4(),
        identity=AgentIdentity(
            "person",
            ConversationKey("channel", "area", "channel", "person"),
        ),
        limits=AgentRunLimits(),
        enabled_tools=("load_agent_skill", "read_agent_skill_resource"),
        skill_scope=scope,
    )


@pytest.mark.asyncio
async def test_skill_scope_charges_parallel_duplicate_activation_once() -> None:
    skill = make_skill()
    scope = make_scope((skill,))

    results = await asyncio.gather(*(scope.load(skill.id) for _ in range(20)))

    assert sum(not result.already_loaded for result in results) == 1
    assert sum(result.returned_characters for result in results) == len(skill.instructions)
    assert scope.activation_count == 1
    assert scope.returned_characters == len(skill.instructions)


@pytest.mark.asyncio
async def test_skill_scope_enforces_visibility_activation_resource_and_context_limits() -> None:
    web = make_skill(instructions="12345", resource_content="abcdef")
    music = make_skill("music-curator")
    scope = make_scope(
        (web, music),
        available=(web,),
        max_activations=1,
        max_resources=1,
        max_instruction_characters=5,
        max_resource_characters=6,
        max_context_characters=11,
    )

    with pytest.raises(AgentSkillScopeError, match="skill_not_available"):
        await scope.load(uuid4())
    with pytest.raises(AgentSkillScopeError, match="skill_not_available"):
        await scope.load(music.id)
    with pytest.raises(AgentSkillScopeError, match="skill_not_activated"):
        await scope.read_resource(web.id, web.resources[0].id)

    activated = await scope.load(web.id)
    with pytest.raises(AgentSkillScopeError, match="skill_resource_not_found"):
        await scope.read_resource(web.id, music.resources[0].id)
    loaded = await scope.read_resource(web.id, web.resources[0].id)
    repeated = await scope.read_resource(web.id, web.resources[0].id)

    assert activated.returned_characters == 5
    assert loaded.returned_characters == 6
    assert repeated.already_loaded is True
    assert repeated.returned_characters == 0
    assert scope.returned_characters == 11


@pytest.mark.asyncio
async def test_skill_scope_rejects_single_item_and_total_character_overflow() -> None:
    oversized = make_skill(instructions="123456")
    single_limit = make_scope(
        (oversized,),
        max_instruction_characters=5,
    )
    with pytest.raises(AgentSkillScopeError, match="skill_context_limit"):
        await single_limit.load(oversized.id)

    total = make_skill(instructions="12345", resource_content="abcdef")
    total_limit = make_scope(
        (total,),
        max_instruction_characters=5,
        max_resource_characters=6,
        max_context_characters=10,
    )
    await total_limit.load(total.id)
    with pytest.raises(AgentSkillScopeError, match="skill_context_limit"):
        await total_limit.read_resource(total.id, total.resources[0].id)


@pytest.mark.asyncio
async def test_skill_scope_enforces_distinct_activation_and_resource_counts() -> None:
    web = make_skill()
    music = make_skill("music-curator")
    activation_scope = make_scope(
        (web, music),
        max_activations=1,
    )
    await activation_scope.load(web.id)
    with pytest.raises(AgentSkillScopeError, match="skill_activation_limit"):
        await activation_scope.load(music.id)

    second_resource = replace(
        web.resources[0],
        id=uuid4(),
        key="second-guide",
        position=2,
    )
    two_resources = replace(web, resources=(web.resources[0], second_resource))
    resource_scope = make_scope(
        (two_resources,),
        max_resources=1,
    )
    await resource_scope.load(two_resources.id)
    await resource_scope.read_resource(two_resources.id, two_resources.resources[0].id)
    with pytest.raises(AgentSkillScopeError, match="skill_resource_limit"):
        await resource_scope.read_resource(
            two_resources.id,
            two_resources.resources[1].id,
        )


@pytest.mark.asyncio
async def test_skill_scope_pins_revision_and_legacy_name_requires_unique_match() -> None:
    first = make_skill()
    duplicate = replace(first, id=uuid4())
    repository = InMemorySkillReadRepository((first, duplicate))
    scope = make_scope((first, duplicate), repository=repository)

    with pytest.raises(AgentSkillScopeError, match="skill_selector_ambiguous"):
        scope.resolve_selector(skill_id=None, deprecated_name=first.name)
    assert scope.resolve_selector(skill_id=first.id, deprecated_name=None) == first.id

    repository.skills[first.id] = replace(first, revision=2)
    with pytest.raises(AgentSkillScopeError, match="skill_revision_changed"):
        await scope.load(first.id)


@pytest.mark.asyncio
async def test_skill_tools_execute_through_registry_policy_and_persist_stable_errors() -> None:
    skill = make_skill()
    scope = make_scope((skill,))
    repository = InMemoryExecutionRepository()
    executor = ToolExecutor(
        ToolRegistry((LoadAgentSkillTool(), ReadAgentSkillResourceTool())),
        ToolPolicy(),
        repository,
    )
    context = execution_context(scope)

    loaded = await executor.execute(
        ToolCall("load", "load_agent_skill", {"skill_id": str(skill.id)}),
        context,
    )
    resource = await executor.execute(
        ToolCall(
            "resource",
            "read_agent_skill_resource",
            {
                "skill_id": str(skill.id),
                "resource_id": str(skill.resources[0].id),
            },
        ),
        context,
    )
    unavailable = await executor.execute(
        ToolCall("unavailable", "load_agent_skill", {"skill_id": str(skill.id)}),
        execution_context(None),
    )

    assert loaded.model_payload()["data"]["instructions"] == skill.instructions
    assert loaded.model_payload()["data"]["skill"]["skill_id"] == str(skill.id)
    assert loaded.model_payload()["data"]["skill"]["access"] == "builtin"
    assert "content" not in loaded.model_payload()["data"]["resources"][0]
    assert resource.model_payload()["data"]["content"] == skill.resources[0].content
    assert unavailable.error_code == "skill_catalog_unavailable"
    assert repository.records[(context.run_id, "load")].status is ToolExecutionStatus.SUCCEEDED
    assert LoadAgentSkillTool().descriptor.replay_in_history is False
    assert ReadAgentSkillResourceTool().descriptor.replay_in_history is False


@pytest.mark.asyncio
async def test_skill_authoring_tool_redacts_long_input_and_returns_compact_result() -> None:
    instructions = "规划步骤。" * 500
    created = replace(
        make_skill(instructions=instructions),
        ownership_kind=SkillOwnershipKind.PERSONAL,
        owner_person_id="person",
        resources=(),
    )

    class Library:
        async def create(self, person_id, **values):
            assert person_id == "person"
            assert values["instructions"] == instructions
            return created

    tool = CreateAgentSkillTool(Library())
    repository = InMemoryExecutionRepository()
    executor = ToolExecutor(ToolRegistry((tool,)), ToolPolicy(), repository)
    context = replace(
        execution_context(None),
        enabled_tools=("create_agent_skill",),
    )

    result = await executor.execute(
        ToolCall(
            "create",
            "create_agent_skill",
            {
                "name": created.name,
                "display_name": created.display_name,
                "description": created.description,
                "instructions": instructions,
                "required_tools": [],
            },
        ),
        context,
    )

    assert result.status is ToolExecutionStatus.SUCCEEDED
    assert result.model_payload()["data"]["skill"]["skill_id"] == str(created.id)
    assert "instructions" not in result.model_payload()["data"]
    execution = repository.records[(context.run_id, "create")]
    assert execution.input_payload["redacted"] is True
    assert instructions not in repr(execution.input_payload)
    assert tool.descriptor.persist_input_payload is False
    assert tool.descriptor.replay_in_history is False


@pytest.mark.asyncio
async def test_skill_inspection_exposes_manifest_and_only_requested_resource_content() -> None:
    skill = make_skill()
    discovery = _discovery(skill)
    manifest = AgentSkillResourceManifest(
        id=skill.resources[0].id,
        key=skill.resources[0].key,
        display_name=skill.resources[0].display_name,
        description=skill.resources[0].description,
        kind=skill.resources[0].kind,
        media_type=skill.resources[0].media_type,
        position=skill.resources[0].position,
    )
    calls: list[str | None] = []

    class Library:
        async def inspect(self, person_id, skill_id, resource_key=None):
            assert (person_id, skill_id) == ("person", skill.id)
            calls.append(resource_key)
            return AgentSkillInspection(
                AgentSkillBundle(discovery, skill.instructions, (manifest,)),
                active=True,
                resource=skill.resources[0] if resource_key is not None else None,
            )

    tool = InspectAgentSkillTool(Library())
    context = execution_context(None)
    without_content = await tool.execute(
        context,
        InspectAgentSkillInput(skill_id=skill.id),
    )
    with_content = await tool.execute(
        context,
        InspectAgentSkillInput(
            skill_id=skill.id,
            resource_key=skill.resources[0].key,
        ),
    )

    assert calls == [None, skill.resources[0].key]
    assert without_content.model_dump()["resource"] is None
    assert "content" not in without_content.model_dump()["resources"][0]
    assert with_content.model_dump()["resource"]["content"] == skill.resources[0].content


@pytest.mark.asyncio
async def test_skill_share_tool_uses_identity_mentions_and_never_accepts_recipient_ids() -> None:
    skill = replace(
        make_skill(),
        ownership_kind=SkillOwnershipKind.PERSONAL,
        owner_person_id="person",
        resources=(),
    )
    now = datetime.now(UTC)
    share = AgentSkillShare(
        id=uuid4(),
        skill_id=skill.id,
        recipient_person_id="friend",
        status=SkillShareStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    calls: list[tuple[str, UUID, tuple[str, ...]]] = []

    class Library:
        async def invite(self, person_id, skill_id, mentioned_person_ids):
            calls.append((person_id, skill_id, mentioned_person_ids))
            return AgentSkillInviteResult(
                replace(_discovery(skill), access=SkillAccessKind.OWNED),
                (share,),
                0,
            )

    tool = InviteAgentSkillShareTool(Library())
    context = replace(
        execution_context(None),
        identity=replace(
            execution_context(None).identity,
            mentioned_person_ids=("friend",),
        ),
    )

    result = await tool.execute(
        context,
        InviteAgentSkillShareInput(skill_id=skill.id),
    )

    assert calls == [("person", skill.id, ("friend",))]
    assert result.invitation_count == 1
    assert "friend" not in repr(result.model_dump())
    with pytest.raises(ValueError):
        InviteAgentSkillShareInput.model_validate(
            {"skill_id": str(skill.id), "recipient_person_id": "attacker-controlled"}
        )


def _discovery(skill: AgentSkill) -> AgentSkillDiscovery:
    return AgentSkillDiscovery(
        id=skill.id,
        name=skill.name,
        display_name=skill.display_name,
        description=skill.description,
        version=skill.version,
        revision=skill.revision,
        required_tools=skill.required_tools,
        access=SkillAccessKind.BUILTIN,
    )
