"""Typed read-only tools exposing run-pinned Agent skill text."""

from __future__ import annotations

import logging
from uuid import UUID

from pydantic import BaseModel, Field

from cywl_oopz.core.lifecycle import ToolEffect
from cywl_oopz.core.observability import exception_kind
from cywl_oopz.features.agent.tools.models import (
    ToolDescriptor,
    ToolExecutionContext,
    ToolExecutionError,
)

from .models import AgentSkill, AgentSkillResource, SkillResourceKind
from .scope import AgentSkillScopeError

logger = logging.getLogger(__name__)

SKILL_TOOL_NAMES = frozenset(
    {
        "load_agent_skill",
        "read_agent_skill_resource",
    }
)


class LoadAgentSkillInput(BaseModel):
    """Select one visible Skill by its stable catalog name."""

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")


class ReadAgentSkillResourceInput(BaseModel):
    """Select one resource from a Skill already activated in this run."""

    skill_name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    resource_id: UUID


class AgentSkillIdentityOutput(BaseModel):
    name: str
    display_name: str
    version: str
    revision: int


class AgentSkillResourceManifestOutput(BaseModel):
    id: UUID
    key: str
    display_name: str
    description: str
    kind: SkillResourceKind


class LoadAgentSkillOutput(BaseModel):
    skill: AgentSkillIdentityOutput
    already_loaded: bool
    instructions: str = ""
    resources: tuple[AgentSkillResourceManifestOutput, ...] = ()
    character_count: int = 0


class ReadAgentSkillResourceOutput(BaseModel):
    skill: AgentSkillIdentityOutput
    resource: AgentSkillResourceManifestOutput
    already_loaded: bool
    media_type: str
    content: str = ""
    character_count: int = 0


class LoadAgentSkillTool:
    """Return full instructions for one Skill visible in the current run."""

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name="load_agent_skill",
            display_name="加载技能",
            description=(
                "按稳定名称加载一个当前可用技能的完整工作说明。"
                "仅当用户任务与技能目录明显匹配或用户明确点名时调用。"
            ),
            input_model=LoadAgentSkillInput,
            output_model=LoadAgentSkillOutput,
            effect=ToolEffect.READ,
            timeout_seconds=2,
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
        if not isinstance(arguments, LoadAgentSkillInput):
            raise TypeError("LoadAgentSkillTool received unexpected arguments")
        scope = context.skill_scope
        if scope is None:
            raise ToolExecutionError("skill_catalog_unavailable")
        try:
            activation = await scope.load(arguments.name)
        except AgentSkillScopeError as exc:
            raise ToolExecutionError(exc.error_code) from exc
        except Exception as exc:
            logger.error(
                "Agent skill activation failed unexpectedly: run=%s skill=%s error=%s",
                context.run_id,
                arguments.name,
                exception_kind(exc),
            )
            raise ToolExecutionError("skill_load_failed") from exc
        skill = activation.skill
        await context.report_progress(
            subject=skill.display_name,
            summary=f"正在载入 v{skill.version}",
        )
        logger.info(
            "Agent skill activated: run=%s skill=%s version=%s revision=%s "
            "characters=%s repeated=%s",
            context.run_id,
            skill.name,
            skill.version,
            skill.revision,
            activation.returned_characters,
            activation.already_loaded,
        )
        return LoadAgentSkillOutput(
            skill=_skill_identity(skill),
            already_loaded=activation.already_loaded,
            instructions="" if activation.already_loaded else skill.instructions,
            resources=(
                ()
                if activation.already_loaded
                else tuple(_resource_manifest(resource) for resource in skill.resources)
            ),
            character_count=activation.returned_characters,
        )


class ReadAgentSkillResourceTool:
    """Return one bounded text resource from an already activated Skill."""

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            name="read_agent_skill_resource",
            display_name="读取技能资料",
            description=(
                "读取已加载技能列出的一份额外文本资料。"
                "必须使用 load_agent_skill 返回的 resource ID。"
            ),
            input_model=ReadAgentSkillResourceInput,
            output_model=ReadAgentSkillResourceOutput,
            effect=ToolEffect.READ,
            timeout_seconds=2,
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
        if not isinstance(arguments, ReadAgentSkillResourceInput):
            raise TypeError("ReadAgentSkillResourceTool received unexpected arguments")
        scope = context.skill_scope
        if scope is None:
            raise ToolExecutionError("skill_catalog_unavailable")
        try:
            loaded = await scope.read_resource(
                arguments.skill_name,
                arguments.resource_id,
            )
        except AgentSkillScopeError as exc:
            raise ToolExecutionError(exc.error_code) from exc
        except Exception as exc:
            logger.error(
                "Agent skill resource read failed unexpectedly: run=%s skill=%s error=%s",
                context.run_id,
                arguments.skill_name,
                exception_kind(exc),
            )
            raise ToolExecutionError("skill_load_failed") from exc
        await context.report_progress(
            subject=loaded.resource.display_name,
            summary=f"正在读取 {loaded.resource.kind.value}",
        )
        logger.info(
            "Agent skill resource read: run=%s skill=%s resource=%s characters=%s repeated=%s",
            context.run_id,
            loaded.skill.name,
            loaded.resource.key,
            loaded.returned_characters,
            loaded.already_loaded,
        )
        return ReadAgentSkillResourceOutput(
            skill=_skill_identity(loaded.skill),
            resource=_resource_manifest(loaded.resource),
            already_loaded=loaded.already_loaded,
            media_type=loaded.resource.media_type,
            content="" if loaded.already_loaded else loaded.resource.content,
            character_count=loaded.returned_characters,
        )


def _skill_identity(skill: AgentSkill) -> AgentSkillIdentityOutput:
    return AgentSkillIdentityOutput(
        name=skill.name,
        display_name=skill.display_name,
        version=skill.version,
        revision=skill.revision,
    )


def _resource_manifest(resource: AgentSkillResource) -> AgentSkillResourceManifestOutput:
    return AgentSkillResourceManifestOutput(
        id=resource.id,
        key=resource.key,
        display_name=resource.display_name,
        description=resource.description,
        kind=resource.kind,
    )
