"""Typed Agent tools for conversational Skill library maintenance."""

from __future__ import annotations

from collections.abc import Awaitable
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.lifecycle import ToolEffect
from cywl_oopz.features.agent.tools.models import (
    ToolDescriptor,
    ToolExecutionContext,
    ToolExecutionError,
)

from .errors import (
    AgentSkillConflictError,
    AgentSkillLibraryError,
    AgentSkillNotFoundError,
    AgentSkillRevisionConflictError,
)
from .library import AgentSkillLibraryService
from .models import (
    AgentSkill,
    AgentSkillDiscovery,
    AgentSkillResource,
    AgentSkillResourceManifest,
    SkillAccessKind,
    SkillResourceKind,
)

SKILL_LIBRARY_TOOL_NAMES = frozenset(
    {
        "list_agent_skill_library",
        "inspect_agent_skill",
        "create_agent_skill",
        "update_agent_skill",
        "manage_agent_skill_resource",
        "set_agent_skill_state",
    }
)


class SkillLibraryInput(BaseModel):
    """Strict base: identity and undeclared controls are never accepted."""

    model_config = ConfigDict(extra="forbid")


class ListAgentSkillLibraryInput(SkillLibraryInput):
    """No model-controlled identity or filter is accepted."""


class InspectAgentSkillInput(SkillLibraryInput):
    skill_id: UUID
    resource_key: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9-]{0,159}$",
    )


class CreateAgentSkillInput(SkillLibraryInput):
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    display_name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1024)
    instructions: str = Field(min_length=1)
    required_tools: frozenset[str] = frozenset()
    version: str = Field(default="1", min_length=1, max_length=64)


class UpdateAgentSkillInput(SkillLibraryInput):
    skill_id: UUID
    expected_revision: int = Field(gt=0)
    name: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9-]{0,63}$",
    )
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, min_length=1, max_length=1024)
    instructions: str | None = Field(default=None, min_length=1)
    required_tools: frozenset[str] | None = None
    version: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_changes(self) -> UpdateAgentSkillInput:
        values = (
            self.name,
            self.display_name,
            self.description,
            self.instructions,
            self.required_tools,
            self.version,
        )
        if all(value is None for value in values):
            raise ValueError("At least one Skill field must change")
        return self


class SkillResourceAction(StrEnum):
    UPSERT = "upsert"
    REMOVE = "remove"


class ManageAgentSkillResourceInput(SkillLibraryInput):
    skill_id: UUID
    expected_revision: int = Field(gt=0)
    action: SkillResourceAction
    key: str = Field(pattern=r"^[a-z][a-z0-9-]{0,159}$")
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, min_length=1, max_length=500)
    kind: SkillResourceKind | None = None
    media_type: str | None = None
    content: str | None = Field(default=None, min_length=1)
    position: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_action_fields(self) -> ManageAgentSkillResourceInput:
        details = (
            self.display_name,
            self.description,
            self.kind,
            self.media_type,
            self.content,
            self.position,
        )
        if self.action is SkillResourceAction.UPSERT and any(value is None for value in details):
            raise ValueError("Resource upsert requires every resource field")
        if self.action is SkillResourceAction.REMOVE and any(
            value is not None for value in details
        ):
            raise ValueError("Resource remove accepts only the resource key")
        return self


class SkillStateAction(StrEnum):
    ARCHIVE = "archive"
    RESTORE = "restore"


class SetAgentSkillStateInput(SkillLibraryInput):
    skill_id: UUID
    expected_revision: int = Field(gt=0)
    action: SkillStateAction


class AgentSkillSummaryOutput(BaseModel):
    skill_id: UUID
    name: str
    display_name: str
    version: str
    revision: int
    access: SkillAccessKind
    active: bool


class AgentSkillResourceManifestOutput(BaseModel):
    id: UUID
    key: str
    display_name: str
    description: str
    kind: SkillResourceKind
    media_type: str
    position: int


class AgentSkillResourceContentOutput(AgentSkillResourceManifestOutput):
    content: str


class ListAgentSkillLibraryOutput(BaseModel):
    owned: tuple[AgentSkillSummaryOutput, ...]
    builtin: tuple[AgentSkillSummaryOutput, ...]
    shared: tuple[AgentSkillSummaryOutput, ...]


class InspectAgentSkillOutput(BaseModel):
    skill: AgentSkillSummaryOutput
    description: str
    instructions: str
    required_tools: tuple[str, ...]
    resources: tuple[AgentSkillResourceManifestOutput, ...]
    resource: AgentSkillResourceContentOutput | None = None


class MutateAgentSkillOutput(BaseModel):
    skill: AgentSkillSummaryOutput
    changed_fields: tuple[str, ...]
    instruction_characters: int
    resource_count: int
    next_run: bool = True


class ListAgentSkillLibraryTool:
    def __init__(self, service: AgentSkillLibraryService) -> None:
        self._service = service
        self._descriptor = ToolDescriptor(
            name="list_agent_skill_library",
            display_name="检查技能库",
            description="列出当前用户拥有、内置和已接受共享的技能，不读取正文。",
            input_model=ListAgentSkillLibraryInput,
            output_model=ListAgentSkillLibraryOutput,
            effect=ToolEffect.READ,
            timeout_seconds=3,
            max_output_characters=16_384,
            concurrency_safe=True,
            idempotent=True,
            replay_in_history=False,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        if not isinstance(arguments, ListAgentSkillLibraryInput):
            raise TypeError("ListAgentSkillLibraryTool received unexpected arguments")
        library = await _library_call(self._service.library(context.identity.person_id))
        return ListAgentSkillLibraryOutput(
            owned=tuple(_summary(item.discovery, active=item.active) for item in library.owned),
            builtin=tuple(_summary(item, active=True) for item in library.builtin),
            shared=tuple(_summary(item, active=True) for item in library.shared),
        )


class InspectAgentSkillTool:
    def __init__(self, service: AgentSkillLibraryService) -> None:
        self._service = service
        self._descriptor = ToolDescriptor(
            name="inspect_agent_skill",
            display_name="查看技能",
            description=(
                "按 skill_id 读取最新技能说明和资料清单；编辑前必须调用。"
                "仅在需要正文时提供 resource_key。"
            ),
            input_model=InspectAgentSkillInput,
            output_model=InspectAgentSkillOutput,
            effect=ToolEffect.READ,
            timeout_seconds=3,
            max_output_characters=32_768,
            concurrency_safe=True,
            idempotent=True,
            replay_in_history=False,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        if not isinstance(arguments, InspectAgentSkillInput):
            raise TypeError("InspectAgentSkillTool received unexpected arguments")
        inspection = await _library_call(
            self._service.inspect(
                context.identity.person_id,
                arguments.skill_id,
                arguments.resource_key,
            )
        )
        bundle = inspection.bundle
        return InspectAgentSkillOutput(
            skill=_summary(bundle.discovery, active=inspection.active),
            description=bundle.discovery.description,
            instructions=bundle.instructions,
            required_tools=tuple(sorted(bundle.discovery.required_tools)),
            resources=tuple(_manifest(resource) for resource in bundle.resources),
            resource=(
                _resource_content(inspection.resource) if inspection.resource is not None else None
            ),
        )


class CreateAgentSkillTool:
    def __init__(self, service: AgentSkillLibraryService) -> None:
        self._service = service
        self._descriptor = _write_descriptor(
            name="create_agent_skill",
            display_name="创建技能",
            description="为当前用户创建一个私有技能；只在用户明确要求创建时调用。",
            input_model=CreateAgentSkillInput,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        if not isinstance(arguments, CreateAgentSkillInput):
            raise TypeError("CreateAgentSkillTool received unexpected arguments")
        skill = await _library_call(
            self._service.create(
                context.identity.person_id,
                name=arguments.name,
                display_name=arguments.display_name,
                description=arguments.description,
                instructions=arguments.instructions,
                required_tools=arguments.required_tools,
                version=arguments.version,
            )
        )
        return _mutation(skill, ("created",))


class UpdateAgentSkillTool:
    def __init__(self, service: AgentSkillLibraryService) -> None:
        self._service = service
        self._descriptor = _write_descriptor(
            name="update_agent_skill",
            display_name="更新技能",
            description=(
                "按 fresh inspect 返回的 revision 更新当前用户拥有的技能；"
                "只在用户明确要求修改时调用。"
            ),
            input_model=UpdateAgentSkillInput,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        if not isinstance(arguments, UpdateAgentSkillInput):
            raise TypeError("UpdateAgentSkillTool received unexpected arguments")
        values = arguments.model_dump(exclude={"skill_id", "expected_revision"})
        changed_fields = tuple(key for key, value in values.items() if value is not None)
        skill = await _library_call(
            self._service.update(
                context.identity.person_id,
                arguments.skill_id,
                arguments.expected_revision,
                **values,
            )
        )
        return _mutation(skill, changed_fields)


class ManageAgentSkillResourceTool:
    def __init__(self, service: AgentSkillLibraryService) -> None:
        self._service = service
        self._descriptor = _write_descriptor(
            name="manage_agent_skill_resource",
            display_name="维护技能资料",
            description=("为当前用户拥有的技能新增、替换或删除一份按需读取的文本资料。"),
            input_model=ManageAgentSkillResourceInput,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        if not isinstance(arguments, ManageAgentSkillResourceInput):
            raise TypeError("ManageAgentSkillResourceTool received unexpected arguments")
        if arguments.action is SkillResourceAction.REMOVE:
            skill = await _library_call(
                self._service.remove_resource(
                    context.identity.person_id,
                    arguments.skill_id,
                    arguments.expected_revision,
                    arguments.key,
                )
            )
            return _mutation(skill, ("resource_removed",))
        skill = await _library_call(
            self._service.upsert_resource(
                context.identity.person_id,
                arguments.skill_id,
                arguments.expected_revision,
                key=arguments.key,
                display_name=arguments.display_name or "",
                description=arguments.description or "",
                kind=arguments.kind or SkillResourceKind.REFERENCE,
                media_type=arguments.media_type or "",
                content=arguments.content or "",
                position=arguments.position or 0,
            )
        )
        return _mutation(skill, ("resource_upserted",))


class SetAgentSkillStateTool:
    def __init__(self, service: AgentSkillLibraryService) -> None:
        self._service = service
        self._descriptor = _write_descriptor(
            name="set_agent_skill_state",
            display_name="设置技能状态",
            description="归档或恢复当前用户拥有的技能；归档前应获得用户明确确认。",
            input_model=SetAgentSkillStateInput,
            persist_input_payload=True,
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    async def execute(
        self,
        context: ToolExecutionContext,
        arguments: BaseModel,
    ) -> BaseModel:
        if not isinstance(arguments, SetAgentSkillStateInput):
            raise TypeError("SetAgentSkillStateTool received unexpected arguments")
        skill = await _library_call(
            self._service.set_state(
                context.identity.person_id,
                arguments.skill_id,
                arguments.expected_revision,
                active=arguments.action is SkillStateAction.RESTORE,
            )
        )
        return _mutation(skill, (arguments.action.value,))


def skill_library_tools(
    service: AgentSkillLibraryService,
) -> tuple[
    ListAgentSkillLibraryTool,
    InspectAgentSkillTool,
    CreateAgentSkillTool,
    UpdateAgentSkillTool,
    ManageAgentSkillResourceTool,
    SetAgentSkillStateTool,
]:
    """Build the complete U2 library tool group for the composition root."""
    return (
        ListAgentSkillLibraryTool(service),
        InspectAgentSkillTool(service),
        CreateAgentSkillTool(service),
        UpdateAgentSkillTool(service),
        ManageAgentSkillResourceTool(service),
        SetAgentSkillStateTool(service),
    )


def _write_descriptor(
    *,
    name: str,
    display_name: str,
    description: str,
    input_model: type[BaseModel],
    persist_input_payload: bool = False,
) -> ToolDescriptor:
    return ToolDescriptor(
        name=name,
        display_name=display_name,
        description=description,
        input_model=input_model,
        output_model=MutateAgentSkillOutput,
        effect=ToolEffect.WRITE,
        timeout_seconds=3,
        max_retries=0,
        max_output_characters=4096,
        concurrency_safe=False,
        idempotent=True,
        replay_in_history=False,
        persist_input_payload=persist_input_payload,
    )


def _summary(
    discovery: AgentSkillDiscovery,
    *,
    active: bool,
) -> AgentSkillSummaryOutput:
    return AgentSkillSummaryOutput(
        skill_id=discovery.id,
        name=discovery.name,
        display_name=discovery.display_name,
        version=discovery.version,
        revision=discovery.revision,
        access=discovery.access,
        active=active,
    )


def _owned_summary(skill: AgentSkill) -> AgentSkillSummaryOutput:
    return AgentSkillSummaryOutput(
        skill_id=skill.id,
        name=skill.name,
        display_name=skill.display_name,
        version=skill.version,
        revision=skill.revision,
        access=SkillAccessKind.OWNED,
        active=skill.enabled and skill.archived_at is None,
    )


def _manifest(
    resource: AgentSkillResourceManifest,
) -> AgentSkillResourceManifestOutput:
    return AgentSkillResourceManifestOutput(
        id=resource.id,
        key=resource.key,
        display_name=resource.display_name,
        description=resource.description,
        kind=resource.kind,
        media_type=resource.media_type,
        position=resource.position,
    )


def _resource_content(
    resource: AgentSkillResource,
) -> AgentSkillResourceContentOutput:
    return AgentSkillResourceContentOutput(
        id=resource.id,
        key=resource.key,
        display_name=resource.display_name,
        description=resource.description,
        kind=resource.kind,
        media_type=resource.media_type,
        position=resource.position,
        content=resource.content,
    )


def _mutation(
    skill: AgentSkill,
    changed_fields: tuple[str, ...],
) -> MutateAgentSkillOutput:
    return MutateAgentSkillOutput(
        skill=_owned_summary(skill),
        changed_fields=changed_fields,
        instruction_characters=len(skill.instructions),
        resource_count=len(skill.resources),
    )


async def _library_call[T](operation: Awaitable[T]) -> T:
    try:
        return await operation
    except AgentSkillLibraryError as exc:
        raise ToolExecutionError(exc.error_code) from exc
    except AgentSkillRevisionConflictError as exc:
        raise ToolExecutionError("skill_revision_conflict") from exc
    except AgentSkillNotFoundError as exc:
        raise ToolExecutionError("skill_not_owned") from exc
    except AgentSkillConflictError as exc:
        raise ToolExecutionError("skill_conflict") from exc
    except DatabaseError as exc:
        raise ToolExecutionError("skill_library_unavailable") from exc
    except ValueError as exc:
        raise ToolExecutionError("invalid_agent_skill") from exc
