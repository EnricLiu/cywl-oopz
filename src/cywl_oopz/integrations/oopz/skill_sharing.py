"""Best-effort OOPZ private notifications for Skill sharing."""

from __future__ import annotations

import logging
from typing import Any

from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.agent.skills.models import AgentSkillDiscovery

logger = logging.getLogger(__name__)


class OopzSkillShareNotifier:
    """Send compact private notices after share transactions commit."""

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    async def invitation(
        self,
        recipient_person_id: str,
        skill: AgentSkillDiscovery,
    ) -> bool:
        return await self._send(
            recipient_person_id,
            (
                f"🎁 **技能邀请** {skill.display_name}\n"
                "有人向你分享了一项只读实时技能。"
                "可使用 /skills invitations 查看，再让未来接受或拒绝。"
            ),
            operation="invitation",
            skill_id=skill.id,
        )

    async def revoked(
        self,
        recipient_person_id: str,
        skill: AgentSkillDiscovery,
    ) -> bool:
        return await self._send(
            recipient_person_id,
            f"ℹ️ **技能分享已撤销** {skill.display_name}",
            operation="revoke",
            skill_id=skill.id,
        )

    async def _send(
        self,
        recipient_person_id: str,
        text: str,
        *,
        operation: str,
        skill_id: object,
    ) -> bool:
        try:
            await self._bot.messages.send_private_message(
                text,
                target=recipient_person_id,
            )
        except Exception as exc:
            logger.warning(
                "OOPZ Skill share notification failed: operation=%s recipient=%s skill=%s error=%s",
                operation,
                opaque_ref(recipient_person_id),
                skill_id,
                exception_kind(exc),
            )
            return False
        return True
