"""User-facing identity and role-management commands."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    AccessRequirement,
    CommandDefinition,
    CommandUsageError,
    NoArguments,
    NoArgumentsParser,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.models import CommandRequest, CommandScope
from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import opaque_ref

from .administration import RoleAdministrationService
from .models import (
    AccessPrincipal,
    AccessResource,
    AccessRole,
    Permission,
    RoleBinding,
    RoleBindingScope,
)
from .policy import RolePermissionPolicy
from .service import AuthorizationService

logger = logging.getLogger(__name__)


class WhoAmICommand:
    """Show the exact OOPZ sender ID used by RBAC."""

    name = "whoami"
    description = "查看权限系统使用的本人 OOPZ ID。"
    category = "权限与管理"
    usage = ("whoami",)

    def definition(self) -> CommandDefinition[NoArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            NoArgumentsParser(),
            self,
            PublicCommandAuthorization(),
        )

    async def handle(self, request: CommandRequest, arguments: NoArguments) -> None:
        del arguments
        await request.responder.reply(f"你的 OOPZ ID：{request.actor.person_id}")


class RoleAction(StrEnum):
    ME = "me"
    LIST = "list"
    GRANT = "grant"
    REVOKE = "revoke"


@dataclass(frozen=True, slots=True)
class RoleMeArguments:
    action: RoleAction = RoleAction.ME


@dataclass(frozen=True, slots=True)
class RoleListArguments:
    subject: AccessPrincipal | None
    action: RoleAction = RoleAction.LIST


@dataclass(frozen=True, slots=True)
class RoleMutationArguments:
    action: RoleAction
    subject: AccessPrincipal
    role: AccessRole
    scope: RoleBindingScope


type RoleArguments = RoleMeArguments | RoleListArguments | RoleMutationArguments


class RoleArgumentsParser:
    """Parse operation, structured mention, role and scope exactly once."""

    _mention_argument = re.compile(r"\(met\)[^\s()]+\(met\)", re.IGNORECASE)

    def parse(self, request: CommandRequest) -> RoleArguments:
        assert request.text is not None
        arguments = tuple(self._mention_argument.sub(" ", request.text.raw_tail).split())
        mentioned = tuple(
            dict.fromkeys(mention.person_id for mention in request.mentions if not mention.is_bot)
        )
        operation = arguments[0].casefold() if arguments else ""
        if operation == RoleAction.ME:
            if len(arguments) != 1 or mentioned:
                raise CommandUsageError("")
            return RoleMeArguments()
        if operation == RoleAction.LIST:
            if len(arguments) != 1:
                raise CommandUsageError("")
            if len(mentioned) > 1:
                raise CommandUsageError(
                    "一次只能查看一位用户。",
                    include_usage=False,
                )
            return RoleListArguments(AccessPrincipal(mentioned[0]) if mentioned else None)
        if operation not in {RoleAction.GRANT, RoleAction.REVOKE}:
            raise CommandUsageError("")
        if len(arguments) != 3:
            raise CommandUsageError("")
        if len(mentioned) != 1:
            raise CommandUsageError(
                "请在当前消息中准确 @ 一位目标用户。",
                include_usage=False,
            )
        try:
            role = AccessRole(arguments[1].casefold())
            scope = RoleBindingScope(arguments[2].casefold())
        except ValueError as exc:
            raise CommandUsageError("") from exc
        if request.location.scope is CommandScope.PRIVATE and scope is not RoleBindingScope.GLOBAL:
            raise CommandUsageError(
                "Area/channel 角色只能在文字频道中管理。",
                include_usage=False,
            )
        return RoleMutationArguments(
            RoleAction(operation),
            AccessPrincipal(mentioned[0]),
            role,
            scope,
        )


class RoleCommandAuthorization:
    """Authorize the exact immutable Role arguments later executed by the handler."""

    def is_available(self, request: CommandRequest) -> bool:
        del request
        return True

    def requirement(
        self,
        request: CommandRequest,
        arguments: RoleArguments,
    ) -> AccessRequirement | None:
        resource = _request_resource(request)
        if isinstance(arguments, RoleMeArguments):
            return None
        if isinstance(arguments, RoleListArguments):
            return AccessRequirement(Permission.RBAC_VIEW, resource)
        return AccessRequirement(
            Permission.RBAC_MANAGE,
            RoleAdministrationService.resource_for_scope(arguments.scope, resource),
        )

    def visibility_requirement(self, request: CommandRequest) -> None:
        del request
        return None


class RoleCommand:
    """Inspect and manage scoped role bindings through real OOPZ mentions."""

    name = "role"
    description = "查看或管理 Bot 角色权限。"
    category = "权限与管理"
    usage = (
        "role me",
        "role list [@用户]",
        "role grant @用户 <角色> <global|area|channel>",
        "role revoke @用户 <角色> <global|area|channel>",
    )
    examples = ("role me", "role grant @用户 admin area")
    max_visible_bindings = 30

    def __init__(
        self,
        authorizer: AuthorizationService,
        administration: RoleAdministrationService,
    ) -> None:
        self._authorizer = authorizer
        self._administration = administration

    def definition(self) -> CommandDefinition[RoleArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            RoleArgumentsParser(),
            self,
            RoleCommandAuthorization(),
        )

    async def handle(self, request: CommandRequest, arguments: RoleArguments) -> None:
        try:
            if isinstance(arguments, RoleMeArguments):
                await self._handle_me(request)
            elif isinstance(arguments, RoleListArguments):
                await self._handle_list(request, arguments)
            else:
                await self._handle_mutation(request, arguments)
        except DatabaseError as exc:
            logger.warning("Role command persistence failed: error=%s", type(exc).__name__)
            await request.responder.reply("权限服务暂时不可用，请稍后重试。")
        except ValueError as exc:
            logger.info("Role command rejected invalid input: error=%s", type(exc).__name__)
            await request.responder.reply(self._input_error(exc))

    async def _handle_me(self, request: CommandRequest) -> None:
        principal = AccessPrincipal(request.actor.person_id)
        resource = _request_resource(request)
        roles = await self._authorizer.effective_roles(principal, resource)
        if not roles:
            await request.responder.reply("当前身份：普通成员\n当前范围内没有管理权限。")
            return
        permissions = frozenset(
            permission for role in roles for permission in RolePermissionPolicy.permissions(role)
        )
        role_names = "、".join(role.value for role in sorted(roles, key=lambda item: item.value))
        permission_names = "、".join(
            permission.value for permission in sorted(permissions, key=lambda item: item.value)
        )
        source = " · bootstrap owner" if self._authorizer.is_bootstrap_owner(principal) else ""
        await request.responder.reply(
            f"当前角色：{role_names}{source}\n"
            f"当前范围：{self._resource_label_value(resource)}\n"
            f"权限：{permission_names}"
        )

    async def _handle_list(
        self,
        request: CommandRequest,
        arguments: RoleListArguments,
    ) -> None:
        records = await self._administration.visible_bindings(
            AccessPrincipal(request.actor.person_id),
            _request_resource(request),
            subject=arguments.subject,
        )
        if not records:
            await request.responder.reply("当前范围内没有可见的角色绑定。")
            return
        visible = records[: self.max_visible_bindings]
        lines = ["**角色绑定**"]
        lines.extend(self._binding_line(record) for record in visible)
        hidden = len(records) - len(visible)
        if hidden:
            lines.append(f"… 另有 {hidden} 项未显示")
        await request.responder.reply("\n".join(lines))

    async def _handle_mutation(
        self,
        request: CommandRequest,
        arguments: RoleMutationArguments,
    ) -> None:
        actor = AccessPrincipal(request.actor.person_id)
        resource = _request_resource(request)
        if arguments.action is RoleAction.GRANT:
            changed = await self._administration.grant(
                actor,
                arguments.subject,
                arguments.role,
                arguments.scope,
                resource,
            )
            verb = "已授予" if changed else "已经拥有"
        else:
            changed = await self._administration.revoke(
                arguments.subject,
                arguments.role,
                arguments.scope,
                resource,
            )
            verb = "已撤销" if changed else "原本没有"
        logger.info(
            "Role binding mutation completed: operation=%s actor=%s subject=%s role=%s "
            "scope=%s changed=%s",
            arguments.action.value,
            opaque_ref(actor.person_id),
            opaque_ref(arguments.subject.person_id),
            arguments.role.value,
            arguments.scope.value,
            changed,
        )
        await request.responder.reply(f"{verb}：{arguments.role.value} · {arguments.scope.value}")

    @staticmethod
    def _binding_line(binding: RoleBinding) -> str:
        address = binding.scope.value
        if binding.scope is RoleBindingScope.AREA:
            address += f":{binding.area_id}"
        elif binding.scope is RoleBindingScope.CHANNEL:
            address += f":{binding.area_id}/{binding.channel_id}"
        return f"• {binding.subject_person_id} · {binding.role.value} · {address}"

    @staticmethod
    def _resource_label_value(resource: AccessResource) -> str:
        if resource.area_id and resource.channel_id:
            return f"channel:{resource.area_id}/{resource.channel_id}"
        return resource.kind.value

    @classmethod
    def _input_error(cls, error: ValueError) -> str:
        if str(error) == "Bootstrap owner roles cannot be revoked":
            return "Bootstrap owner 不能通过命令撤销，请修改本地环境配置。"
        if str(error) == "Area and channel roles require a channel context":
            return "Area/channel 角色只能在文字频道中管理。"
        if str(error) == "Channel roles require a channel context":
            return "Channel 角色只能在具体文字频道中管理。"
        return "角色操作失败，请检查命令参数。"


def _request_resource(request: CommandRequest) -> AccessResource:
    if request.location.scope is CommandScope.PRIVATE:
        return AccessResource.private()
    return AccessResource.channel(
        request.location.area_id,
        request.location.channel_id,
    )
