"""Chat-specific naming for the application task supervisor."""

from __future__ import annotations

from cywl_oopz.core.tasks import TaskSupervisor
from cywl_oopz.features.admin.models import OopzMessageScope, OutboundMessageReceipt

from .models import ConversationKey


class ChatTaskSupervisor(TaskSupervisor[ConversationKey]):
    """Own at most one LLM reply task per conversation."""

    def __init__(self) -> None:
        super().__init__(lambda key: f"chat:{key.scope}")


class OutboundChatTaskCanceller:
    """Project adapter from an outbound receipt to its conversation task key."""

    def __init__(self, tasks: ChatTaskSupervisor) -> None:
        self._tasks = tasks

    async def cancel_for_message(self, receipt: OutboundMessageReceipt) -> bool:
        person_id = receipt.owner_person_id
        if not person_id:
            return False
        address = receipt.address
        key = ConversationKey(
            scope=address.scope.value,
            area_id=address.area_id if address.scope is OopzMessageScope.CHANNEL else "",
            channel_id=address.channel_id if address.scope is OopzMessageScope.CHANNEL else "",
            person_id=person_id,
        )
        return await self._tasks.cancel(key)
