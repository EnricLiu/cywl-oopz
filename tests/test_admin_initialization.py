from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from cywl_oopz.commands.builtin import HelpCommand
from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.features.access.models import (
    AccessRole,
    RoleBinding,
    RoleBindingScope,
)
from cywl_oopz.features.access.service import AuthorizationService
from cywl_oopz.features.admin.commands import InitCommand
from cywl_oopz.features.admin.initialization import (
    ChannelCatalogError,
    ChannelInitializationService,
)
from cywl_oopz.features.admin.models import (
    AreaChannelCatalog,
    AreaInitializationResult,
    ChannelInitializationResult,
    ChannelKey,
)
from cywl_oopz.integrations.oopz.channel_catalog import OopzAreaChannelCatalog


@dataclass
class InMemoryRoleBindings:
    records: list[RoleBinding] = field(default_factory=list)

    async def list_for_subject(self, subject_person_id: str) -> tuple[RoleBinding, ...]:
        return tuple(
            record for record in self.records if record.subject_person_id == subject_person_id
        )


class FakeCatalog:
    def __init__(self, catalog: AreaChannelCatalog) -> None:
        self.catalog = catalog
        self.calls: list[str] = []

    async def discover(self, area_id: str) -> AreaChannelCatalog:
        self.calls.append(area_id)
        return self.catalog


@dataclass
class InMemoryInitializationRepository:
    text: set[ChannelKey] = field(default_factory=set)
    voice: set[ChannelKey] = field(default_factory=set)
    text_configuration: dict[ChannelKey, str] = field(default_factory=dict)

    async def initialize_text_channel(
        self,
        channel: ChannelKey,
    ) -> ChannelInitializationResult:
        created = channel not in self.text
        self.text.add(channel)
        self.text_configuration.setdefault(channel, "database-default")
        return ChannelInitializationResult(created)

    async def initialize_area(
        self,
        catalog: AreaChannelCatalog,
    ) -> AreaInitializationResult:
        text_created = sum(channel not in self.text for channel in catalog.text_channels)
        voice_created = sum(channel not in self.voice for channel in catalog.voice_channels)
        for channel in catalog.text_channels:
            self.text.add(channel)
            self.text_configuration.setdefault(channel, "database-default")
        self.voice.update(catalog.voice_channels)
        return AreaInitializationResult(
            text_created,
            len(catalog.text_channels) - text_created,
            voice_created,
            len(catalog.voice_channels) - voice_created,
        )


class FakeMessage:
    def __init__(self, text: str, person_id: str = "person") -> None:
        self.plain_text = text
        self.text = text
        self.content = text
        self.sender_id = person_id
        self.area = "area"
        self.channel = "text"


class FakeContext:
    def __init__(self, message: FakeMessage, *, private: bool = False) -> None:
        self.event = SimpleNamespace(message=message, is_private=private)
        self.replies: list[str] = []

    async def reply(self, text: str) -> None:
        self.replies.append(text)


def make_router(
    roles: InMemoryRoleBindings,
    repository: InMemoryInitializationRepository,
    catalog: FakeCatalog,
    *,
    bootstrap: frozenset[str] = frozenset(),
) -> CommandRouter:
    authorizer = AuthorizationService(roles, bootstrap)
    router = CommandRouter("/", authorizer)
    router.register_definition(
        InitCommand(ChannelInitializationService(catalog, repository)).definition()
    )
    return router


@pytest.mark.asyncio
async def test_channel_admin_initializes_once_without_overwriting_existing_configuration() -> None:
    key = ChannelKey("area", "text")
    roles = InMemoryRoleBindings(
        [
            RoleBinding(
                "person",
                AccessRole.ADMIN,
                RoleBindingScope.CHANNEL,
                area_id="area",
                channel_id="text",
            )
        ]
    )
    repository = InMemoryInitializationRepository()
    catalog = FakeCatalog(AreaChannelCatalog("area"))
    router = make_router(roles, repository, catalog)

    first = FakeContext(FakeMessage("/init"))
    assert await router.dispatch(first.event.message, first)
    assert "频道已初始化" in first.replies[0]

    repository.text_configuration[key] = "custom"
    second = FakeContext(FakeMessage("/init channel"))
    assert await router.dispatch(second.event.message, second)
    assert second.replies == ["频道已经初始化，现有配置未改动。"]
    assert repository.text_configuration[key] == "custom"


@pytest.mark.asyncio
async def test_channel_scoped_admin_cannot_initialize_an_area() -> None:
    roles = InMemoryRoleBindings(
        [
            RoleBinding(
                "person",
                AccessRole.ADMIN,
                RoleBindingScope.CHANNEL,
                area_id="area",
                channel_id="text",
            )
        ]
    )
    repository = InMemoryInitializationRepository()
    catalog = FakeCatalog(AreaChannelCatalog("area"))
    router = make_router(roles, repository, catalog)
    context = FakeContext(FakeMessage("/init area"))

    assert await router.dispatch(context.event.message, context)

    assert context.replies == ["你没有执行此操作的权限。"]
    assert catalog.calls == []


@pytest.mark.asyncio
async def test_area_admin_initializes_visible_text_and_voice_channels() -> None:
    roles = InMemoryRoleBindings(
        [RoleBinding("person", AccessRole.ADMIN, RoleBindingScope.AREA, area_id="area")]
    )
    repository = InMemoryInitializationRepository(
        text={ChannelKey("area", "text-existing")},
        voice={ChannelKey("area", "voice-existing")},
    )
    catalog = FakeCatalog(
        AreaChannelCatalog(
            "area",
            (
                ChannelKey("area", "text-existing"),
                ChannelKey("area", "text-new"),
            ),
            (
                ChannelKey("area", "voice-existing"),
                ChannelKey("area", "voice-new"),
            ),
        )
    )
    router = make_router(roles, repository, catalog)
    context = FakeContext(FakeMessage("/init area"))

    assert await router.dispatch(context.event.message, context)

    assert "文字频道：新增 1 · 已存在 1" in context.replies[0]
    assert "语音频道：新增 1 · 已存在 1" in context.replies[0]
    assert catalog.calls == ["area"]


@pytest.mark.asyncio
async def test_init_is_hidden_and_rejected_in_private_conversations() -> None:
    roles = InMemoryRoleBindings()
    repository = InMemoryInitializationRepository()
    catalog = FakeCatalog(AreaChannelCatalog("area"))
    router = make_router(roles, repository, catalog, bootstrap=frozenset({"owner"}))
    router.register(HelpCommand(router))
    help_context = FakeContext(FakeMessage("/help", "owner"), private=True)

    assert await router.dispatch(help_context.event.message, help_context)
    assert "/init" not in help_context.replies[0]

    init_context = FakeContext(FakeMessage("/init", "owner"), private=True)
    assert await router.dispatch(init_context.event.message, init_context)
    assert init_context.replies == ["此命令只能在文字频道中使用。"]


@pytest.mark.asyncio
async def test_oopz_catalog_flattens_deduplicates_and_classifies_audio_as_voice() -> None:
    groups = [
        SimpleNamespace(
            channels=[
                SimpleNamespace(channel_id="text", channel_type="TEXT"),
                SimpleNamespace(channel_id="voice", channel_type="VOICE"),
                SimpleNamespace(channel_id="legacy", channel_type="AUDIO"),
            ]
        ),
        SimpleNamespace(
            channels=[
                SimpleNamespace(channel_id="text", channel_type="TEXT"),
                SimpleNamespace(channel_id="ignored", channel_type="VIDEO"),
                SimpleNamespace(channel_id="", channel_type="TEXT"),
            ]
        ),
    ]

    class Areas:
        async def get_area_channels(self, area_id: str):
            assert area_id == "area"
            return groups

    adapter = OopzAreaChannelCatalog(SimpleNamespace(areas=Areas()))

    result = await adapter.discover("area")

    assert result.text_channels == (ChannelKey("area", "text"),)
    assert result.voice_channels == (
        ChannelKey("area", "voice"),
        ChannelKey("area", "legacy"),
    )


@pytest.mark.asyncio
async def test_oopz_catalog_timeout_is_translated_to_domain_error() -> None:
    class Areas:
        async def get_area_channels(self, area_id: str):
            del area_id
            await asyncio.Event().wait()

    adapter = OopzAreaChannelCatalog(
        SimpleNamespace(areas=Areas()),
        timeout_seconds=0.001,
    )

    with pytest.raises(ChannelCatalogError):
        await adapter.discover("area")
