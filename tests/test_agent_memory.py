from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from types import SimpleNamespace
from uuid import UUID

import pytest

from cywl_oopz.commands.router import ParsedCommand
from cywl_oopz.features.agent.commands import MemoryCommand
from cywl_oopz.features.agent.memory import MemoryItem, MemoryService
from cywl_oopz.settings import AgentSettings


class InMemoryRepository:
    def __init__(self) -> None:
        self.preferences: dict[str, bool] = {}
        self.items: dict[UUID, MemoryItem] = {}

    async def preference(self, person_id: str) -> bool | None:
        return self.preferences.get(person_id)

    async def set_preference(self, person_id: str, enabled: bool) -> None:
        self.preferences[person_id] = enabled

    async def add(self, item: MemoryItem) -> None:
        self.items[item.id] = item

    async def list_active(
        self,
        person_id: str,
        now: datetime,
        *,
        limit: int,
    ) -> tuple[MemoryItem, ...]:
        return tuple(
            item
            for item in self.items.values()
            if item.owner_person_id == person_id
            and (item.expires_at is None or item.expires_at > now)
        )[:limit]

    async def count_active(self, person_id: str, now: datetime) -> int:
        return len(await self.list_active(person_id, now, limit=10_000))

    async def delete(self, person_id: str, item_id: UUID) -> bool:
        item = self.items.get(item_id)
        if item is None or item.owner_person_id != person_id:
            return False
        del self.items[item_id]
        return True

    async def delete_all(self, person_id: str) -> int:
        owned = [
            item_id for item_id, item in self.items.items() if item.owner_person_id == person_id
        ]
        for item_id in owned:
            del self.items[item_id]
        return len(owned)

    async def touch(
        self,
        person_id: str,
        item_ids: tuple[UUID, ...],
        now: datetime,
    ) -> None:
        for item_id in item_ids:
            item = self.items.get(item_id)
            if item is not None and item.owner_person_id == person_id:
                self.items[item_id] = replace(item, last_used_at=now)


@dataclass
class FakeContext:
    replies: list[str] = field(default_factory=list)
    event: object = field(
        default_factory=lambda: SimpleNamespace(
            is_private=True,
            message=SimpleNamespace(sender_id="person"),
        )
    )

    async def reply(self, text: str) -> None:
        self.replies.append(text)


def settings() -> AgentSettings:
    return AgentSettings.from_mapping(
        {
            "CYWL_AGENT_MEMORY_DEFAULT_TTL_DAYS": "30",
            "CYWL_AGENT_MEMORY_MAX_ITEMS": "2",
            "CYWL_AGENT_MEMORY_CONTEXT_ITEMS": "2",
            "CYWL_AGENT_MEMORY_MAX_ITEM_CHARACTERS": "40",
        }
    )


@pytest.mark.asyncio
async def test_memory_service_is_owner_scoped_expiring_and_user_controlled() -> None:
    repository = InMemoryRepository()
    service = MemoryService(settings(), repository)

    item = await service.remember("person", "我喜欢爵士乐")
    await service.remember("other", "另一个用户的资料")
    context = await service.context_text("person")
    other_delete = await service.forget("other", item.id)
    await service.set_enabled("person", False)

    assert context == "- 我喜欢爵士乐"
    assert repository.items[item.id].last_used_at is not None
    assert other_delete is False
    assert await service.context_text("person") == ""
    assert len(await service.list("person")) == 1
    assert (await service.status("person")).enabled is False


@pytest.mark.asyncio
async def test_memory_service_enforces_item_and_size_limits() -> None:
    service = MemoryService(settings(), InMemoryRepository())

    await service.remember("person", "first")
    await service.remember("person", "second")

    with pytest.raises(ValueError, match="limit"):
        await service.remember("person", "third")
    with pytest.raises(ValueError, match="too long"):
        await MemoryService(
            settings(),
            InMemoryRepository(),
        ).remember("person", "x" * 41)


@pytest.mark.asyncio
async def test_memory_command_remember_list_off_and_forget_all() -> None:
    memory = MemoryService(settings(), InMemoryRepository())
    command = MemoryCommand(SimpleNamespace(), memory)
    remember_context = FakeContext()
    list_context = FakeContext()
    off_context = FakeContext()
    forget_context = FakeContext()

    await command.execute(
        ParsedCommand("memory", ("remember", "喜欢", "Lo-fi")),
        remember_context,
    )
    await command.execute(ParsedCommand("memory", ("list",)), list_context)
    await command.execute(ParsedCommand("memory", ("off",)), off_context)
    await command.execute(
        ParsedCommand("memory", ("forget", "all")),
        forget_context,
    )

    assert "记忆 ID" in remember_context.replies[0]
    assert "喜欢 Lo-fi" in list_context.replies[0]
    assert "已关闭" in off_context.replies[0]
    assert forget_context.replies == ["已删除 1 条长期记忆。"]

    disabled_context = FakeContext()
    await command.execute(
        ParsedCommand("memory", ("remember", "new")),
        disabled_context,
    )
    assert disabled_context.replies == ["长期记忆当前已关闭；请先使用 !memory on。"]
