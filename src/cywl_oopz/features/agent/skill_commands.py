"""Typed Agent Skill discovery and personal-library command."""

from __future__ import annotations

from enum import StrEnum

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    CommandDefinition,
    CommandExecutionPolicy,
    CommandUsageError,
    ExecutionMode,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.models import CommandRequest

from .command_support import AgentCommandContext, AgentCommandErrorPresenter
from .service import AgentConversationService
from .skills.library import AgentSkillLibraryService


class SkillsAction(StrEnum):
    AVAILABLE = "available"
    OWNED = "owned"
    SHARED = "shared"
    INVITATIONS = "invitations"


class SkillsArgumentsParser:
    def parse(self, request: CommandRequest) -> SkillsAction:
        assert request.text is not None
        tokens = request.text.tokens
        if not tokens:
            return SkillsAction.AVAILABLE
        if len(tokens) == 1 and tokens[0].casefold() in {
            "owned",
            "shared",
            "invitations",
        }:
            try:
                return SkillsAction(tokens[0].casefold())
            except ValueError:
                pass
        raise CommandUsageError(
            "",
            include_usage=True,
        )


class SkillsCommand:
    name = "skills"
    description = "查看当前 Agent 可按需加载的技能。"
    category = "Agent"
    usage = ("skills [owned|shared|invitations]",)
    max_reply_characters = 1900

    def __init__(
        self,
        service: AgentConversationService,
        library: AgentSkillLibraryService | None = None,
    ) -> None:
        self._agent = service
        self._library = library
        self._errors = AgentCommandErrorPresenter()

    def definition(self) -> CommandDefinition[SkillsAction]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            SkillsArgumentsParser(),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=30.0),
        )

    async def handle(self, request: CommandRequest, arguments: SkillsAction) -> None:
        if arguments is not SkillsAction.AVAILABLE:
            await self._show_library_group(arguments, request)
            return
        try:
            skills = await self._agent.available_skills(AgentCommandContext.conversation(request))
        except Exception as exc:
            await self._errors.reply(request, exc)
            return
        if not skills:
            await request.responder.reply("当前对话没有可用的 Agent 技能。")
            return
        access_labels = {"builtin": "内置", "owned": "我的", "shared": "共享"}
        lines = ["当前可用 Agent 技能（需要时由模型按需加载）："]
        for index, skill in enumerate(skills):
            description = (
                skill.description
                if len(skill.description) <= 240
                else f"{skill.description[:237]}..."
            )
            access = access_labels[skill.access.value]
            line = (
                f"- **{skill.display_name}** {skill.name} · {access} · "
                f"v{skill.version}：{description}"
            )
            omitted = len(skills) - index
            suffix = f"\n…另有 {omitted} 个技能未显示。" if omitted else ""
            if len("\n".join((*lines, line))) + len(suffix) > self.max_reply_characters:
                lines.append(f"…另有 {omitted} 个技能未显示。")
                break
            lines.append(line)
        await request.responder.reply("\n".join(lines))

    async def _show_library_group(
        self,
        action: SkillsAction,
        request: CommandRequest,
    ) -> None:
        if self._library is None:
            await request.responder.reply("当前未启用个人技能库。")
            return
        try:
            library = await self._library.library(request.actor.person_id)
        except Exception as exc:
            await self._errors.reply(request, exc)
            return
        if action is SkillsAction.OWNED:
            lines = ["**我的技能**"]
            if not library.owned:
                lines.append("（暂无）")
            else:
                lines.extend(
                    f"- **{item.discovery.display_name}** {item.discovery.name} · "
                    f"{'使用中' if item.active else '已归档'}"
                    for item in library.owned
                )
        elif action is SkillsAction.SHARED:
            lines = ["**已接受的共享技能**"]
            if not library.shared:
                lines.append("（暂无）")
            else:
                lines.extend(
                    f"- **{skill.display_name}** {skill.name} · v{skill.version}"
                    for skill in library.shared
                )
        else:
            lines = ["**待处理的技能邀请**"]
            if not library.pending_invitations:
                lines.append("（暂无）")
            else:
                lines.extend(
                    f"- **{item.skill.display_name}** · 邀请 ID {item.share.id}"
                    f"{'' if item.active else ' · 当前已归档'}"
                    for item in library.pending_invitations
                )
                lines.append("可直接告诉未来接受或拒绝其中一项。")
        await request.responder.reply(self._bounded(lines))

    def _bounded(self, lines: list[str]) -> str:
        rendered: list[str] = []
        for index, line in enumerate(lines):
            remaining = len(lines) - index
            omitted = f"\n…另有 {remaining} 项未显示。" if remaining else ""
            if len("\n".join((*rendered, line))) + len(omitted) > self.max_reply_characters:
                rendered.append(f"…另有 {remaining} 项未显示。")
                break
            rendered.append(line)
        return "\n".join(rendered)
