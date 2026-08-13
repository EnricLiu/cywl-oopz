"""User-facing identity and role-management commands."""

from __future__ import annotations

import logging

from oopz_sdk.events.context import EventContext

from cywl_oopz.commands.router import AccessRequirement, ParsedCommand
from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import opaque_ref
from cywl_oopz.integrations.oopz.access import OopzAccessInvocation

from .administration import RoleAdministrationService
from .models import AccessPrincipal, AccessRole, Permission, RoleBinding, RoleBindingScope
from .policy import RolePermissionPolicy
from .service import AuthorizationService

logger = logging.getLogger(__name__)


class RoleCommandAccess:
    """Resolve the mixed public/view/manage paths of `/role`."""

    def requirement(
        self,
        command: ParsedCommand,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement | None:
        if not command.arguments or command.arguments[0].casefold() == "me":
            return None
        operation = command.arguments[0].casefold()
        if operation == "list":
            return AccessRequirement(Permission.RBAC_VIEW, invocation.resource)
        if operation in {"grant", "revoke"}:
            resource = invocation.resource
            if len(command.arguments) >= 3:
                try:
                    scope = RoleBindingScope(command.arguments[2].casefold())
                    resource = RoleAdministrationService.resource_for_scope(
                        scope, invocation.resource
                    )
                except ValueError:
                    resource = invocation.resource
            return AccessRequirement(Permission.RBAC_MANAGE, resource)
        return None

    def visibility_requirement(
        self,
        invocation: OopzAccessInvocation,
    ) -> AccessRequirement | None:
        del invocation
        return None


class WhoAmICommand:
    """Show the exact OOPZ sender ID used by RBAC."""

    name = "whoami"
    description = "查看权限系统使用的本人 OOPZ ID。"

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        if command.arguments:
            await context.reply("用法：/whoami")
            return
        invocation = OopzAccessInvocation.from_context(context)
        await context.reply(f"你的 OOPZ ID：{invocation.principal.person_id}")


class RoleCommand:
    """Inspect and manage scoped role bindings through real OOPZ mentions."""

    name = "role"
    description = "查看或管理 Bot 角色权限。"
    max_visible_bindings = 30

    def __init__(
        self,
        authorizer: AuthorizationService,
        administration: RoleAdministrationService,
    ) -> None:
        self._authorizer = authorizer
        self._administration = administration

    async def execute(self, command: ParsedCommand, context: EventContext) -> None:
        operation = command.arguments[0].casefold() if command.arguments else ""
        try:
            if operation == "me":
                await self._me(command, context)
            elif operation == "list":
                await self._list(command, context)
            elif operation in {"grant", "revoke"}:
                await self._mutate(operation, command, context)
            else:
                await context.reply(self._usage())
        except DatabaseError as exc:
            logger.warning("Role command persistence failed: error=%s", type(exc).__name__)
            await context.reply("权限服务暂时不可用，请稍后重试。")
        except ValueError as exc:
            logger.info("Role command rejected invalid input: error=%s", type(exc).__name__)
            await context.reply(self._input_error(exc))

    async def _me(self, command: ParsedCommand, context: EventContext) -> None:
        if len(command.arguments) != 1 or self._mentioned_people(context):
            raise ValueError("invalid_role_me")
        invocation = OopzAccessInvocation.from_context(context)
        roles = await self._authorizer.effective_roles(invocation.principal, invocation.resource)
        if not roles:
            await context.reply("当前身份：普通成员\n当前范围内没有管理权限。")
            return
        permissions = frozenset(
            permission for role in roles for permission in RolePermissionPolicy.permissions(role)
        )
        role_names = "、".join(role.value for role in sorted(roles, key=lambda item: item.value))
        permission_names = "、".join(
            permission.value for permission in sorted(permissions, key=lambda item: item.value)
        )
        source = (
            " · bootstrap owner"
            if self._authorizer.is_bootstrap_owner(invocation.principal)
            else ""
        )
        await context.reply(
            f"当前角色：{role_names}{source}\n"
            f"当前范围：{self._resource_label(invocation)}\n"
            f"权限：{permission_names}"
        )

    async def _list(self, command: ParsedCommand, context: EventContext) -> None:
        if len(command.arguments) != 1:
            raise ValueError("invalid_role_list")
        mentions = self._mentioned_people(context)
        if len(mentions) > 1:
            raise ValueError("role_target_limit")
        invocation = OopzAccessInvocation.from_context(context)
        records = await self._administration.visible_bindings(
            invocation.principal,
            invocation.resource,
            subject=AccessPrincipal(mentions[0]) if mentions else None,
        )
        if not records:
            await context.reply("当前范围内没有可见的角色绑定。")
            return
        visible = records[: self.max_visible_bindings]
        lines = ["**角色绑定**"]
        lines.extend(self._binding_line(record) for record in visible)
        hidden = len(records) - len(visible)
        if hidden:
            lines.append(f"… 另有 {hidden} 项未显示")
        await context.reply("\n".join(lines))

    async def _mutate(
        self,
        operation: str,
        command: ParsedCommand,
        context: EventContext,
    ) -> None:
        if len(command.arguments) != 3:
            raise ValueError("invalid_role_mutation")
        mentions = self._mentioned_people(context)
        if len(mentions) != 1:
            raise ValueError("role_target_required")
        try:
            role = AccessRole(command.arguments[1].casefold())
            scope = RoleBindingScope(command.arguments[2].casefold())
        except ValueError as exc:
            raise ValueError("invalid_role_or_scope") from exc
        invocation = OopzAccessInvocation.from_context(context)
        subject = AccessPrincipal(mentions[0])
        if operation == "grant":
            changed = await self._administration.grant(
                invocation.principal,
                subject,
                role,
                scope,
                invocation.resource,
            )
            verb = "已授予" if changed else "已经拥有"
        else:
            changed = await self._administration.revoke(
                subject,
                role,
                scope,
                invocation.resource,
            )
            verb = "已撤销" if changed else "原本没有"
        logger.info(
            "Role binding mutation completed: operation=%s actor=%s subject=%s role=%s "
            "scope=%s changed=%s",
            operation,
            opaque_ref(invocation.principal.person_id),
            opaque_ref(subject.person_id),
            role.value,
            scope.value,
            changed,
        )
        await context.reply(f"{verb}：{role.value} · {scope.value}")

    @staticmethod
    def _mentioned_people(context: EventContext) -> tuple[str, ...]:
        event = getattr(context, "event", None)
        message = getattr(event, "message", None)
        bot_id = str(getattr(getattr(context, "config", None), "person_uid", "")).strip()
        result: list[str] = []
        for mention in getattr(message, "mention_list", ()) or ():
            person_id = str(getattr(mention, "person", "")).strip()
            if person_id and person_id != bot_id and person_id not in result:
                result.append(person_id)
        return tuple(result)

    @staticmethod
    def _binding_line(binding: RoleBinding) -> str:
        address = binding.scope.value
        if binding.scope is RoleBindingScope.AREA:
            address += f":{binding.area_id}"
        elif binding.scope is RoleBindingScope.CHANNEL:
            address += f":{binding.area_id}/{binding.channel_id}"
        return f"• {binding.subject_person_id} · {binding.role.value} · {address}"

    @staticmethod
    def _resource_label(invocation: OopzAccessInvocation) -> str:
        resource = invocation.resource
        if resource.area_id and resource.channel_id:
            return f"channel:{resource.area_id}/{resource.channel_id}"
        return resource.kind.value

    @staticmethod
    def _usage() -> str:
        return (
            "用法：\n"
            "/role me\n"
            "/role list [@用户]\n"
            "/role grant @用户 <owner|admin|moderator> <global|area|channel>\n"
            "/role revoke @用户 <owner|admin|moderator> <global|area|channel>"
        )

    @classmethod
    def _input_error(cls, error: ValueError) -> str:
        if str(error) == "Bootstrap owner roles cannot be revoked":
            return "Bootstrap owner 不能通过命令撤销，请修改本地环境配置。"
        if str(error) == "Area and channel roles require a channel context":
            return "Area/channel 角色只能在文字频道中管理。"
        if str(error) == "Channel roles require a channel context":
            return "Channel 角色只能在具体文字频道中管理。"
        if str(error) == "role_target_required":
            return "请在当前消息中准确 @ 一位目标用户。"
        if str(error) == "role_target_limit":
            return "一次只能查看一位用户。"
        return cls._usage()
