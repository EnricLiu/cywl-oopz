"""User-controlled long-term memory values and use cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Protocol
from uuid import UUID, uuid4

from cywl_oopz.settings import AgentSettings


class MemoryDisabledError(ValueError):
    """Raised when a user tries to write while their memory is disabled."""


class MemoryCapacityError(ValueError):
    """Raised when a user reaches the configured item count."""


class MemoryItemTooLongError(ValueError):
    """Raised when one explicit memory item exceeds its bounded size."""


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One user-owned, expiring long-term memory item."""

    id: UUID
    owner_person_id: str
    namespace: str
    content: Mapping[str, Any]
    source_thread_id: UUID | None
    source_message_sequence: int | None
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None

    def __post_init__(self) -> None:
        if not self.owner_person_id.strip() or not self.namespace.strip():
            raise ValueError("Memory owner and namespace must not be empty")
        object.__setattr__(self, "content", MappingProxyType(dict(self.content)))


@dataclass(frozen=True, slots=True)
class MemoryStatus:
    """Safe status exposed by the memory command."""

    enabled: bool
    item_count: int


class MemoryRepository(Protocol):
    """Persistence boundary that always scopes reads and deletes by owner."""

    async def preference(self, person_id: str) -> bool | None:
        """Return explicit preference, or None when the application default applies."""

    async def set_preference(self, person_id: str, enabled: bool) -> None:
        """Insert or update one user's memory preference."""

    async def add(self, item: MemoryItem) -> None:
        """Persist one explicit memory item."""

    async def list_active(
        self,
        person_id: str,
        now: datetime,
        *,
        limit: int,
    ) -> tuple[MemoryItem, ...]:
        """Return only this owner's unexpired items."""

    async def count_active(self, person_id: str, now: datetime) -> int:
        """Count only this owner's unexpired items."""

    async def delete(self, person_id: str, item_id: UUID) -> bool:
        """Delete one owned item, returning whether it existed."""

    async def delete_all(self, person_id: str) -> int:
        """Delete all items owned by one user."""

    async def touch(self, person_id: str, item_ids: tuple[UUID, ...], now: datetime) -> None:
        """Update last-used timestamps for items still owned by this user."""


class MemoryService:
    """Implement explicit remember/list/forget/on/off and bounded context projection."""

    def __init__(self, settings: AgentSettings, repository: MemoryRepository) -> None:
        self._settings = settings
        self._repository = repository

    async def status(self, person_id: str) -> MemoryStatus:
        now = datetime.now(UTC)
        return MemoryStatus(
            enabled=await self.is_enabled(person_id),
            item_count=await self._repository.count_active(person_id, now),
        )

    async def is_enabled(self, person_id: str) -> bool:
        explicit = await self._repository.preference(person_id)
        return self._settings.memory_enabled_by_default if explicit is None else explicit

    async def set_enabled(self, person_id: str, enabled: bool) -> None:
        await self._repository.set_preference(person_id, enabled)

    async def remember(self, person_id: str, text: str) -> MemoryItem:
        content = text.strip()
        if not content:
            raise ValueError("Memory text must not be empty")
        if len(content) > self._settings.memory_max_item_characters:
            raise MemoryItemTooLongError("Memory text is too long")
        if not await self.is_enabled(person_id):
            raise MemoryDisabledError("Long-term memory is disabled")
        now = datetime.now(UTC)
        if await self._repository.count_active(person_id, now) >= self._settings.memory_max_items:
            raise MemoryCapacityError("Long-term memory item limit reached")
        item = MemoryItem(
            id=uuid4(),
            owner_person_id=person_id,
            namespace="explicit",
            content={"text": content},
            source_thread_id=None,
            source_message_sequence=None,
            created_at=now,
            updated_at=now,
            last_used_at=None,
            expires_at=now + timedelta(days=self._settings.memory_default_ttl_days),
        )
        await self._repository.add(item)
        return item

    async def list(self, person_id: str) -> tuple[MemoryItem, ...]:
        return await self._repository.list_active(
            person_id,
            datetime.now(UTC),
            limit=self._settings.memory_max_items,
        )

    async def forget(self, person_id: str, item_id: UUID) -> bool:
        return await self._repository.delete(person_id, item_id)

    async def forget_all(self, person_id: str) -> int:
        return await self._repository.delete_all(person_id)

    async def context_text(self, person_id: str) -> str:
        """Project enabled items into a bounded, explicitly untrusted text block."""
        if not await self.is_enabled(person_id):
            return ""
        now = datetime.now(UTC)
        items = await self._repository.list_active(
            person_id,
            now,
            limit=self._settings.memory_context_items,
        )
        if not items:
            return ""
        await self._repository.touch(
            person_id,
            tuple(item.id for item in items),
            now,
        )
        lines: list[str] = []
        for item in items:
            text = item.content.get("text")
            if isinstance(text, str) and text.strip():
                lines.append(f"- {text.strip()}")
        return "\n".join(lines)
