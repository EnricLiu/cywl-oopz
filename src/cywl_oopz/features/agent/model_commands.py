"""Typed Provider and model selection commands."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
from cywl_oopz.core.lifecycle import ModelSelectionSource
from cywl_oopz.features.chat.tasks import ChatTaskSupervisor

from .command_support import AgentCommandContext, AgentCommandErrorPresenter
from .models import ModelSelection, SelectableModel
from .service import AgentConversationService


class ModelCommandView:
    """Render compact OOPZ-safe Provider/model menus with stable command aliases."""

    max_characters = 1900
    _source_labels = {
        ModelSelectionSource.THREAD: "当前对话",
        ModelSelectionSource.USER: "个人默认",
        ModelSelectionSource.CHANNEL: "频道默认",
        ModelSelectionSource.APPLICATION: "系统默认",
    }

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def providers(self, selection: ModelSelection, choices: tuple[SelectableModel, ...]) -> str:
        groups: dict[tuple[str, str], list[SelectableModel]] = defaultdict(list)
        for choice in choices:
            groups[(choice.provider_alias, choice.provider_display_name)].append(choice)
        footer = (
            "",
            f"切换当前对话：{self._prefix}provider <Provider> [模型]",
            f"设为个人默认：{self._prefix}provider default <Provider> [模型]",
        )
        lines = [self._current(selection), "", "**可用 Provider**"]
        for (alias, display_name), models in groups.items():
            model_labels = "、".join(
                f"{model.model_alias}{'（默认）' if model.is_provider_default else ''}"
                for model in models
            )
            line = f"- **{alias}** {self._compact_name(display_name)}：{model_labels}"
            if not self._can_append(lines, line, footer):
                lines.append("…其余 Provider 已省略")
                break
            lines.append(line)
        if not groups:
            lines.append("（当前没有可选模型）")
        return "\n".join((*lines, *footer))

    def models(self, selection: ModelSelection, choices: tuple[SelectableModel, ...]) -> str:
        current_provider = selection.model.provider_alias
        provider_choices = tuple(
            choice for choice in choices if choice.provider_alias == current_provider
        )
        footer = (
            "",
            f"切换模型：{self._prefix}model <模型>",
            f"跨 Provider：{self._prefix}model <Provider>/<模型>",
            f"查看全部 Provider：{self._prefix}provider",
        )
        lines = [self._current(selection), "", f"**{current_provider} 可用模型**"]
        for choice in provider_choices:
            labels = []
            if choice.model_display_name != choice.model_alias:
                labels.append(self._compact_name(choice.model_display_name))
            if choice.is_provider_default:
                labels.append("Provider 默认")
            suffix = f" · {' · '.join(labels)}" if labels else ""
            line = f"- **{choice.model_alias}**{suffix}"
            if not self._can_append(lines, line, footer):
                lines.append("…其余模型已省略")
                break
            lines.append(line)
        if not provider_choices:
            lines.append("（当前 Provider 没有其他可选模型）")
        return "\n".join((*lines, *footer))

    def provider_usage(self) -> str:
        return "\n".join(
            (
                "**Provider 命令**",
                f"- {self._prefix}provider：查看当前选择和全部 Provider",
                f"- {self._prefix}provider <Provider> [模型]：切换当前对话",
                f"- {self._prefix}provider default <Provider> [模型]：设置个人默认",
                f"- 旧写法 {self._prefix}provider use ... 仍然可用",
            )
        )

    def model_usage(self) -> str:
        return "\n".join(
            (
                "**模型命令**",
                f"- {self._prefix}model：查看当前 Provider 的模型",
                f"- {self._prefix}model <模型>：在当前 Provider 内切换",
                f"- {self._prefix}model <Provider>/<模型>：跨 Provider 切换",
            )
        )

    def active_run_message(self) -> str:
        return f"当前正在生成回复；请等待完成或先使用 {self._prefix}cancel。"

    @classmethod
    def _current(cls, selection: ModelSelection) -> str:
        source = cls._source_labels.get(selection.source, selection.source.value)
        return (
            f"🎛️ **当前模型** {selection.model.provider_alias}/"
            f"{selection.model.model_alias} · {source}"
        )

    @staticmethod
    def _compact_name(value: str) -> str:
        return value if len(value) <= 80 else f"{value[:77]}..."

    def _can_append(self, lines: list[str], line: str, footer: tuple[str, ...]) -> bool:
        return len("\n".join((*lines, line, *footer))) <= self.max_characters


class SelectionAction(StrEnum):
    LIST = "list"
    HELP = "help"
    SELECT = "select"


@dataclass(frozen=True, slots=True)
class ModelArguments:
    action: SelectionAction
    choice: str = ""


class ModelArgumentsParser:
    def __init__(self, view: ModelCommandView) -> None:
        self._view = view

    def parse(self, request: CommandRequest) -> ModelArguments:
        assert request.text is not None
        tokens = request.text.tokens
        if not tokens or tokens == ("list",):
            return ModelArguments(SelectionAction.LIST)
        if tokens == ("help",):
            return ModelArguments(SelectionAction.HELP)
        if len(tokens) == 1 and tokens[0].casefold() not in {"help", "list"}:
            return ModelArguments(SelectionAction.SELECT, tokens[0])
        raise CommandUsageError(self._view.model_usage(), include_usage=False)


class AgentModelCommand:
    name = "model"
    description = "查看当前 Provider 的模型，或切换当前对话模型。"
    category = "对话"
    usage = ("model [list|help|模型编号或名称]",)

    def __init__(
        self,
        service: AgentConversationService,
        tasks: ChatTaskSupervisor,
        prefix: str = "/",
    ) -> None:
        self._agent = service
        self._tasks = tasks
        self._view = ModelCommandView(prefix)
        self._errors = AgentCommandErrorPresenter()

    def definition(self) -> CommandDefinition[ModelArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            ModelArgumentsParser(self._view),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=30.0),
        )

    async def handle(self, request: CommandRequest, arguments: ModelArguments) -> None:
        key = AgentCommandContext.conversation(request)
        if arguments.action is SelectionAction.HELP:
            await request.responder.reply(self._view.model_usage())
            return
        if arguments.action is SelectionAction.LIST:
            try:
                catalog = await self._agent.model_catalog_view(key)
            except Exception as exc:
                await self._errors.reply(request, exc)
                return
            await request.responder.reply(self._view.models(catalog.selection, catalog.choices))
            return
        if self._tasks.has_active(key):
            await request.responder.reply(self._view.active_run_message())
            return
        try:
            selected = await self._agent.select_model(key, arguments.choice)
        except ValueError:
            await request.responder.reply(
                f"没有找到可选模型「{arguments.choice}」。\n{self._view.model_usage()}"
            )
            return
        except Exception as exc:
            await self._errors.reply(request, exc)
            return
        await request.responder.reply(f"✅ **当前对话模型** {selected}")


@dataclass(frozen=True, slots=True)
class ProviderArguments:
    action: SelectionAction
    provider_alias: str = ""
    model_alias: str | None = None
    user_default: bool = False


class ProviderArgumentsParser:
    def __init__(self, view: ModelCommandView) -> None:
        self._view = view

    def parse(self, request: CommandRequest) -> ProviderArguments:
        assert request.text is not None
        tokens = request.text.tokens
        if not tokens or tokens == ("list",):
            return ProviderArguments(SelectionAction.LIST)
        if tokens == ("help",):
            return ProviderArguments(SelectionAction.HELP)
        action = tokens[0].casefold()
        if action in {"help", "list"}:
            self._invalid()
        arguments = tokens[1:] if action in {"use", "default"} else tokens
        user_default = action == "default"
        if not 1 <= len(arguments) <= 2:
            self._invalid()
        if len(arguments) == 2:
            provider_alias, model_alias = arguments
        else:
            value = arguments[0]
            if "/" not in value:
                provider_alias, model_alias = value, None
            else:
                provider_alias, model_alias = value.split("/", 1)
                if not provider_alias or not model_alias:
                    self._invalid()
        return ProviderArguments(
            SelectionAction.SELECT,
            provider_alias,
            model_alias,
            user_default,
        )

    def _invalid(self) -> None:
        raise CommandUsageError(self._view.provider_usage(), include_usage=False)


class ProviderCommand:
    name = "provider"
    description = "查看、列出或切换 Agent Provider/模型。"
    category = "Agent"
    usage = (
        "provider [list|help]",
        "provider [use] <Provider> [模型]",
        "provider default <Provider> [模型]",
    )

    def __init__(
        self,
        service: AgentConversationService,
        tasks: ChatTaskSupervisor,
        prefix: str = "/",
    ) -> None:
        self._agent = service
        self._tasks = tasks
        self._view = ModelCommandView(prefix)
        self._errors = AgentCommandErrorPresenter()

    def definition(self) -> CommandDefinition[ProviderArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            ProviderArgumentsParser(self._view),
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=30.0),
        )

    async def handle(self, request: CommandRequest, arguments: ProviderArguments) -> None:
        key = AgentCommandContext.conversation(request)
        if arguments.action is SelectionAction.HELP:
            await request.responder.reply(self._view.provider_usage())
            return
        if arguments.action is SelectionAction.LIST:
            try:
                catalog = await self._agent.model_catalog_view(key)
            except Exception as exc:
                await self._errors.reply(request, exc)
                return
            await request.responder.reply(self._view.providers(catalog.selection, catalog.choices))
            return
        if self._tasks.has_active(key):
            await request.responder.reply(self._view.active_run_message())
            return
        try:
            selected = await self._agent.select_provider(
                key,
                arguments.provider_alias,
                arguments.model_alias,
                user_default=arguments.user_default,
            )
        except ValueError:
            target = arguments.provider_alias + (
                f"/{arguments.model_alias}" if arguments.model_alias else ""
            )
            await request.responder.reply(
                f"没有找到可选的 Provider/模型「{target}」。\n{self._view.provider_usage()}"
            )
            return
        except Exception as exc:
            await self._errors.reply(request, exc)
            return
        if arguments.user_default:
            message = (
                f"✅ **个人默认模型** {selected}\n"
                "未单独选择模型的对话会使用它；当前对话的独立选择不会被覆盖。"
            )
        else:
            message = f"✅ **当前对话模型** {selected}"
        await request.responder.reply(message)
