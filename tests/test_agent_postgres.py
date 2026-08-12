from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from cywl_oopz.features.agent.catalog import ReloadableProviderCatalog
from cywl_oopz.features.agent.memory import MemoryItem
from cywl_oopz.features.agent.memory_repository import SqlAlchemyMemoryRepository
from cywl_oopz.features.agent.models import (
    AgentMessage,
    AgentRun,
    AgentRunLimits,
    AgentRunState,
    AgentRunStatus,
    AgentStopReason,
    AgentThread,
    LlmModel,
    LlmProvider,
    ModelCapability,
    ModelSelectionSource,
    ProviderProtocol,
)
from cywl_oopz.features.agent.repository import (
    SqlAlchemyAgentMessageRepository,
    SqlAlchemyAgentRunRepository,
    SqlAlchemyAgentThreadRepository,
    SqlAlchemyModelSelectionRepository,
    SqlAlchemyProviderCatalogRepository,
    SqlAlchemyToolExecutionRepository,
)
from cywl_oopz.features.agent.selection import ProviderSelectionService
from cywl_oopz.features.agent.skills.errors import (
    AgentSkillConflictError,
    AgentSkillNotFoundError,
    AgentSkillRevisionConflictError,
)
from cywl_oopz.features.agent.skills.models import (
    AgentSkill,
    AgentSkillResource,
    SkillAccessKind,
    SkillOwnershipKind,
    SkillResourceKind,
    SkillShareStatus,
)
from cywl_oopz.features.agent.skills.repository import (
    SqlAlchemyAgentSkillRepository,
)
from cywl_oopz.features.agent.tools.models import (
    ToolEffect,
    ToolExecution,
    ToolExecutionStatus,
)
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.music.errors import (
    MusicPlaylistConflictError,
    MusicPlaylistFullError,
    MusicPlaylistNotFoundError,
)
from cywl_oopz.features.music.models import MusicTrack
from cywl_oopz.features.music.playlist_repository import (
    SqlAlchemyMusicPlaylistRepository,
)
from cywl_oopz.storage.models import (
    AgentRunRecord,
    ChannelSettingsRecord,
    LlmModelRecord,
)
from cywl_oopz.storage.url import normalize_asyncpg_url


@pytest.mark.asyncio
async def test_agent_migration_constraints_and_repositories_on_postgresql() -> None:
    if os.getenv("CYWL_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("set CYWL_RUN_POSTGRES_TESTS=1 to run isolated PostgreSQL tests")

    load_dotenv(find_dotenv(usecwd=True), override=False)
    database_url = normalize_asyncpg_url(os.environ["DATABASE_URL"])
    schema = f"cywl_agent_test_{uuid4().hex}"
    admin_engine = create_async_engine(database_url)
    test_engine = None
    try:
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))

        test_engine = create_async_engine(
            database_url,
            connect_args={"server_settings": {"search_path": schema}},
        )
        await _migrate(test_engine, "upgrade", "20260726_01")
        async with test_engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO channel_settings (
                        id, area_id, channel_id, chat_enabled, created_at, updated_at
                    )
                    VALUES (
                        :id, 'legacy-area', 'legacy-channel', true, now(), now()
                    )
                    """
                ),
                {"id": uuid4()},
            )
        await _migrate(test_engine, "upgrade", "head")
        async with test_engine.connect() as connection:
            catalog_state = await connection.scalar(
                text("SELECT to_regclass('agent_skill_catalog_state')")
            )
            generation_triggers = await connection.scalar(
                text(
                    """
                        SELECT count(*)
                        FROM pg_trigger
                        JOIN pg_class ON pg_class.oid = pg_trigger.tgrelid
                        JOIN pg_namespace ON pg_namespace.oid = pg_class.relnamespace
                        WHERE pg_namespace.nspname = current_schema()
                          AND tgname IN (
                            'trg_agent_skills_bump_generation',
                            'trg_agent_skill_resources_bump_generation'
                        )
                    """
                )
            )
        assert catalog_state is None
        assert generation_triggers == 0
        sessions = async_sessionmaker(test_engine, expire_on_commit=False)
        async with sessions() as session:
            legacy_record = await session.scalar(
                select(ChannelSettingsRecord).where(ChannelSettingsRecord.area_id == "legacy-area")
            )
        assert legacy_record is not None
        assert legacy_record.chat_enabled is True
        assert legacy_record.default_model_id is None
        assert legacy_record.enabled_agent_tools == [
            "get_agent_status",
            "get_channel_settings",
            "react_to_message",
            "search_music_catalog",
            "enqueue_music",
            "get_music_queue",
            "skip_music",
            "pause_music",
            "resume_music",
            "search_web",
            "read_web_page",
            "set_music_playback_mode",
            "create_music_playlist",
            "list_music_playlists",
            "get_music_playlist",
            "add_music_playlist_track",
            "remove_music_playlist_track",
            "load_music_playlist",
            "load_agent_skill",
            "read_agent_skill_resource",
            "preview_netease_playlist",
            "import_netease_playlist",
            "list_agent_skill_library",
            "inspect_agent_skill",
            "create_agent_skill",
            "update_agent_skill",
            "manage_agent_skill_resource",
            "set_agent_skill_state",
            "invite_agent_skill_share",
            "respond_agent_skill_share",
            "revoke_agent_skill_share",
            "clear_music_queue",
            "rename_music_playlist",
            "delete_music_playlist",
            "clear_music_playlist",
        ]

        async with test_engine.begin() as connection:
            defaulted = (
                (
                    await connection.execute(
                        text(
                            """
                        INSERT INTO channel_settings (area_id, channel_id)
                        VALUES ('default-area', 'default-channel')
                        RETURNING
                            id,
                            chat_enabled,
                            enabled_agent_tools,
                            created_at,
                            updated_at
                        """
                        )
                    )
                )
                .mappings()
                .one()
            )
        assert defaulted["id"] is not None
        assert defaulted["chat_enabled"] is False
        assert defaulted["enabled_agent_tools"] == [
            "get_agent_status",
            "get_channel_settings",
            "react_to_message",
            "search_music_catalog",
            "enqueue_music",
            "get_music_queue",
            "skip_music",
            "pause_music",
            "resume_music",
            "clear_music_queue",
            "search_web",
            "read_web_page",
            "browser_open",
            "browser_snapshot",
            "browser_wait",
            "browser_close",
            "browser_click",
            "browser_fill",
            "browser_press",
            "set_music_playback_mode",
            "create_music_playlist",
            "list_music_playlists",
            "get_music_playlist",
            "add_music_playlist_track",
            "remove_music_playlist_track",
            "rename_music_playlist",
            "delete_music_playlist",
            "clear_music_playlist",
            "load_music_playlist",
            "load_agent_skill",
            "read_agent_skill_resource",
            "list_agent_skill_library",
            "inspect_agent_skill",
            "create_agent_skill",
            "update_agent_skill",
            "manage_agent_skill_resource",
            "set_agent_skill_state",
            "invite_agent_skill_share",
            "respond_agent_skill_share",
            "revoke_agent_skill_share",
            "preview_netease_playlist",
            "import_netease_playlist",
        ]
        assert defaulted["created_at"] == defaulted["updated_at"]

        with pytest.raises(DBAPIError):
            async with sessions.begin() as session:
                await session.execute(
                    text(
                        """
                        INSERT INTO agent_skills (
                            name, display_name, description, instructions, version,
                            required_tools
                        )
                        VALUES (
                            'invalid-skill', 'Invalid', 'Invalid duplicate tools',
                            'This row must be rejected.', '1',
                            '["search_web", "search_web"]'::jsonb
                        )
                        """
                    )
                )

        async with test_engine.begin() as connection:
            skill_row = (
                (
                    await connection.execute(
                        text(
                            """
                            INSERT INTO agent_skills (
                                name, display_name, description, instructions, version,
                                required_tools
                            )
                            VALUES (
                                'test-research', '网页研究',
                                '搜索并阅读可靠来源。', '先搜索，再阅读关键原文。', '1',
                                '["search_web", "read_web_page"]'::jsonb
                            )
                            RETURNING id, revision, metadata, enabled, created_at, updated_at
                            """
                        )
                    )
                )
                .mappings()
                .one()
            )
            resource_row = (
                (
                    await connection.execute(
                        text(
                            """
                            INSERT INTO agent_skill_resources (
                                skill_id, key, display_name, description, kind,
                                media_type, content, position
                            )
                            VALUES (
                                :skill_id, 'source-guide', '来源指南',
                                '需要判断来源质量时读取。', 'reference',
                                'text/markdown', '# 来源指南\\n优先选择一手来源。', 1
                            )
                            RETURNING id, created_at, updated_at
                            """
                        ),
                        {"skill_id": skill_row["id"]},
                    )
                )
                .mappings()
                .one()
            )
        assert skill_row["revision"] == 1
        assert skill_row["metadata"] == {}
        assert skill_row["enabled"] is True
        assert skill_row["created_at"] == skill_row["updated_at"]
        assert resource_row["id"] is not None
        assert resource_row["created_at"] == resource_row["updated_at"]

        skill_repository = SqlAlchemyAgentSkillRepository(sessions)
        discoveries = await skill_repository.list_accessible("builtin-reader")
        skills_by_name = {skill.name: skill for skill in discoveries}
        assert set(skills_by_name) == {
            "music-curator",
            "netease-playlist-importer",
            "skill-authoring",
            "test-research",
            "web-research",
        }
        music_skill = skills_by_name["music-curator"]
        import_skill = skills_by_name["netease-playlist-importer"]
        authoring_skill = skills_by_name["skill-authoring"]
        web_skill = skills_by_name["web-research"]
        music_bundle = await skill_repository.load_accessible_bundle(
            "builtin-reader",
            music_skill.id,
            music_skill.revision,
        )
        import_bundle = await skill_repository.load_accessible_bundle(
            "builtin-reader",
            import_skill.id,
            import_skill.revision,
        )
        authoring_bundle = await skill_repository.load_accessible_bundle(
            "builtin-reader",
            authoring_skill.id,
            authoring_skill.revision,
        )
        web_bundle = await skill_repository.load_accessible_bundle(
            "builtin-reader",
            web_skill.id,
            web_skill.revision,
        )
        assert all(
            bundle is not None
            for bundle in (
                music_bundle,
                import_bundle,
                authoring_bundle,
                web_bundle,
            )
        )
        assert music_bundle is not None
        assert import_bundle is not None
        assert authoring_bundle is not None
        assert web_bundle is not None
        assert music_skill.version == "1.2.0"
        assert music_skill.required_tools == frozenset(
            {
                "search_music_catalog",
                "get_music_queue",
                "set_music_playback_mode",
                "list_music_playlists",
                "get_music_playlist",
                "add_music_playlist_track",
                "rename_music_playlist",
                "delete_music_playlist",
                "clear_music_playlist",
                "clear_music_queue",
                "load_music_playlist",
            }
        )
        assert [resource.key for resource in music_bundle.resources] == ["batch-curation-guide"]
        assert "`source=auto`" in music_bundle.instructions
        assert "真实的 `source` 与 `source_id`" in music_bundle.instructions
        assert "共享歌单允许混合来源" in music_bundle.instructions
        assert import_skill.version == "1.0.0"
        assert import_skill.required_tools == frozenset(
            {
                "list_music_playlists",
                "preview_netease_playlist",
                "import_netease_playlist",
            }
        )
        assert [resource.key for resource in import_bundle.resources] == ["netease-api-behavior"]
        assert "分享个人 Skill" in authoring_skill.description
        assert authoring_skill.required_tools == frozenset(
            {
                "list_agent_skill_library",
                "inspect_agent_skill",
                "create_agent_skill",
                "update_agent_skill",
                "manage_agent_skill_resource",
                "set_agent_skill_state",
                "invite_agent_skill_share",
                "respond_agent_skill_share",
                "revoke_agent_skill_share",
            }
        )
        assert "真实 `@` 提及" in authoring_bundle.instructions
        assert web_skill.version == "1.0.0"
        assert web_skill.required_tools == frozenset({"search_web", "read_web_page"})
        assert [resource.key for resource in web_bundle.resources] == ["source-evaluation"]
        loaded_skill = skills_by_name["test-research"]
        loaded_bundle = await skill_repository.load_accessible_bundle(
            "builtin-reader",
            loaded_skill.id,
            loaded_skill.revision,
        )
        assert loaded_bundle is not None
        loaded_resource = await skill_repository.read_accessible_resource(
            "builtin-reader",
            loaded_skill.id,
            loaded_bundle.resources[0].id,
            loaded_skill.revision,
        )
        assert loaded_resource is not None
        assert loaded_skill.revision == 2
        assert loaded_skill.required_tools == frozenset({"search_web", "read_web_page"})
        assert loaded_resource.id == resource_row["id"]
        assert loaded_resource.kind is SkillResourceKind.REFERENCE
        assert loaded_skill.access is SkillAccessKind.BUILTIN

        first_personal = AgentSkill(
            id=uuid4(),
            name="travel-planner",
            display_name="旅行规划",
            description="为 owner 规划旅行。",
            instructions="根据 owner 的目标整理行程。",
            version="1",
            revision=1,
            required_tools=frozenset(),
            resources=(),
            metadata={"preferences": {"seasons": ["spring", "autumn"]}},
            ownership_kind=SkillOwnershipKind.PERSONAL,
            owner_person_id="owner-one",
        )
        second_personal = replace(
            first_personal,
            id=uuid4(),
            display_name="另一位用户的旅行规划",
            owner_person_id="owner-two",
        )
        await skill_repository.add_personal(first_personal)
        await skill_repository.add_personal(second_personal)
        with pytest.raises(AgentSkillConflictError):
            await skill_repository.add_personal(replace(first_personal, id=uuid4()))

        builtin_ids = {
            skill.id for skill in await skill_repository.list_accessible("builtin-reader")
        }
        assert first_personal.id not in builtin_ids
        assert second_personal.id not in builtin_ids
        owner_one_ids = {skill.id for skill in await skill_repository.list_accessible("owner-one")}
        unrelated_ids = {skill.id for skill in await skill_repository.list_accessible("recipient")}
        assert first_personal.id in owner_one_ids
        assert second_personal.id not in owner_one_ids
        assert first_personal.id not in unrelated_ids
        owned_skill = await skill_repository.get_owned(" owner-one ", first_personal.id)
        assert owned_skill is not None
        assert owned_skill.metadata["preferences"] == {"seasons": ("spring", "autumn")}
        assert await skill_repository.get_owned("owner-two", first_personal.id) is None

        invitation_time = datetime.now(UTC)
        invitations = await skill_repository.invite_many(
            "owner-one",
            first_personal.id,
            ("recipient", "recipient-two"),
            invitation_time,
        )
        invitation = invitations[0]
        assert invitation.status is SkillShareStatus.PENDING
        assert [item.recipient_person_id for item in invitations] == [
            "recipient",
            "recipient-two",
        ]
        pending = await skill_repository.pending_invitations("recipient")
        assert [item.share.id for item in pending] == [invitation.id]
        outgoing = await skill_repository.outgoing_shares("owner-one")
        first_outgoing = next(item for item in outgoing if item.skill.id == first_personal.id)
        assert (first_outgoing.pending_count, first_outgoing.accepted_count) == (2, 0)
        assert first_personal.id not in {
            skill.id for skill in await skill_repository.list_accessible("recipient")
        }
        with pytest.raises(AgentSkillNotFoundError):
            await skill_repository.respond(
                "other-recipient",
                invitation.id,
                SkillShareStatus.ACCEPTED,
                invitation_time,
            )
        accepted = await skill_repository.respond(
            "recipient",
            invitation.id,
            SkillShareStatus.ACCEPTED,
            invitation_time,
        )
        assert accepted.status is SkillShareStatus.ACCEPTED
        repeated = await skill_repository.respond(
            "recipient",
            invitation.id,
            SkillShareStatus.ACCEPTED,
            invitation_time,
        )
        assert repeated.status is SkillShareStatus.ACCEPTED
        with pytest.raises(AgentSkillConflictError):
            await skill_repository.respond(
                "recipient",
                invitation.id,
                SkillShareStatus.DECLINED,
                invitation_time,
            )
        summary = await skill_repository.share_for_recipient("recipient", invitation.id)
        assert summary is not None
        assert summary.share.status is SkillShareStatus.ACCEPTED
        outgoing = await skill_repository.outgoing_shares("owner-one")
        first_outgoing = next(item for item in outgoing if item.skill.id == first_personal.id)
        assert (first_outgoing.pending_count, first_outgoing.accepted_count) == (1, 1)
        assert first_personal.id in {
            skill.id for skill in await skill_repository.list_accessible("recipient")
        }
        recipient_discovery = await skill_repository.list_accessible("recipient")
        discovery_by_id = {item.id: item for item in recipient_discovery}
        assert discovery_by_id[first_personal.id].access is SkillAccessKind.SHARED
        assert discovery_by_id[loaded_skill.id].access is SkillAccessKind.BUILTIN
        shared_bundle = await skill_repository.load_accessible_bundle(
            "recipient",
            first_personal.id,
            first_personal.revision,
        )
        assert shared_bundle is not None
        assert shared_bundle.discovery.access is SkillAccessKind.SHARED
        assert shared_bundle.instructions == first_personal.instructions
        assert shared_bundle.resources == ()
        builtin_bundle = await skill_repository.load_accessible_bundle(
            "recipient",
            loaded_skill.id,
            loaded_skill.revision,
        )
        assert builtin_bundle is not None
        assert builtin_bundle.resources[0].id == loaded_resource.id
        builtin_resource = await skill_repository.read_accessible_resource(
            "recipient",
            loaded_skill.id,
            loaded_resource.id,
            loaded_skill.revision,
        )
        assert builtin_resource == loaded_resource
        with pytest.raises(AgentSkillRevisionConflictError):
            await skill_repository.load_accessible_bundle(
                "recipient",
                first_personal.id,
                first_personal.revision + 1,
            )
        assert (
            await skill_repository.load_accessible_bundle(
                "unrelated",
                first_personal.id,
                first_personal.revision,
            )
            is None
        )
        with pytest.raises(AgentSkillNotFoundError):
            await skill_repository.revoke_owned_shares(
                "owner-two",
                first_personal.id,
                ("recipient",),
            )
        removed = await skill_repository.revoke_owned_shares(
            "owner-one",
            first_personal.id,
            ("recipient",),
        )
        assert removed == (accepted,)
        assert first_personal.id not in {
            skill.id for skill in await skill_repository.list_accessible("recipient")
        }
        remaining = await skill_repository.revoke_owned_shares(
            "owner-one",
            first_personal.id,
            None,
        )
        assert [item.recipient_person_id for item in remaining] == ["recipient-two"]
        owned_summaries = await skill_repository.list_owned("owner-one")
        assert [(item.discovery.id, item.active) for item in owned_summaries] == [
            (first_personal.id, True)
        ]
        inspected = await skill_repository.inspect_accessible(
            "owner-one",
            first_personal.id,
        )
        assert inspected is not None
        assert inspected.bundle.instructions == first_personal.instructions
        assert inspected.resource is None

        updated_personal = await skill_repository.update_owned(
            replace(
                first_personal,
                description="为 owner 规划并核对旅行安排。",
            ),
            first_personal.revision,
        )
        assert updated_personal.revision == first_personal.revision + 1
        with pytest.raises(AgentSkillRevisionConflictError):
            await skill_repository.update_owned(
                replace(updated_personal, version="2"),
                first_personal.revision,
            )
        personal_resource = AgentSkillResource(
            id=uuid4(),
            key="packing-list",
            display_name="行李清单",
            description="整理出发前的行李时读取。",
            kind=SkillResourceKind.TEMPLATE,
            media_type="text/markdown",
            content="# 行李清单\n- 证件",
            position=1,
        )
        with_resource = await skill_repository.upsert_owned_resource(
            "owner-one",
            first_personal.id,
            updated_personal.revision,
            personal_resource,
        )
        assert with_resource.revision == updated_personal.revision + 1
        assert with_resource.resources == (personal_resource,)
        inspected_resource = await skill_repository.read_inspectable_resource(
            "owner-one",
            first_personal.id,
            personal_resource.key,
        )
        assert inspected_resource == personal_resource
        without_resource = await skill_repository.remove_owned_resource(
            "owner-one",
            first_personal.id,
            with_resource.revision,
            personal_resource.key,
        )
        assert without_resource.revision == with_resource.revision + 1
        assert without_resource.resources == ()
        archived = await skill_repository.set_owned_state(
            "owner-one",
            first_personal.id,
            without_resource.revision,
            enabled=False,
            archived_at=datetime.now(UTC),
        )
        assert archived.revision == without_resource.revision + 1
        assert first_personal.id not in {
            item.id for item in await skill_repository.list_accessible("owner-one")
        }
        assert (await skill_repository.list_owned("owner-one"))[0].active is False
        restored = await skill_repository.set_owned_state(
            "owner-one",
            first_personal.id,
            archived.revision,
            enabled=True,
            archived_at=None,
        )
        assert restored.revision == archived.revision + 1
        assert first_personal.id in {
            item.id for item in await skill_repository.list_accessible("owner-one")
        }
        before_resource_update = await skill_repository.read_accessible_resource(
            "builtin-reader",
            loaded_skill.id,
            loaded_resource.id,
            loaded_skill.revision,
        )
        assert before_resource_update is not None

        async with test_engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    UPDATE agent_skill_resources
                    SET content = '# 来源指南\\n优先选择官方一手来源。'
                    WHERE id = :resource_id
                    """
                ),
                {"resource_id": resource_row["id"]},
            )
        with pytest.raises(AgentSkillRevisionConflictError):
            await skill_repository.load_accessible_bundle(
                "builtin-reader",
                loaded_skill.id,
                loaded_skill.revision,
            )
        reloaded_skill = {
            skill.name: skill for skill in await skill_repository.list_accessible("builtin-reader")
        }["test-research"]
        after_resource_update = await skill_repository.read_accessible_resource(
            "builtin-reader",
            reloaded_skill.id,
            loaded_resource.id,
            reloaded_skill.revision,
        )
        assert after_resource_update is not None
        assert reloaded_skill.revision == loaded_skill.revision + 1
        assert "官方" not in before_resource_update.content
        assert "官方" in after_resource_update.content

        playlist_repository = SqlAlchemyMusicPlaylistRepository(sessions)
        playlist = await playlist_repository.create(
            "shared-area",
            "夜间 电台",
            "夜间 电台".casefold(),
            "person",
        )
        with pytest.raises(MusicPlaylistConflictError):
            await playlist_repository.create(
                "shared-area",
                "夜间 电台",
                "夜间 电台".casefold(),
                "other",
            )
        first_playlist_track = await playlist_repository.append(
            "shared-area",
            playlist.id,
            MusicTrack("netease", "first", "First", ("Miku",), 1000),
            "person",
            max_tracks=2,
        )
        second_playlist_track = await playlist_repository.append(
            "shared-area",
            playlist.id,
            MusicTrack("netease", "second", "Second", ("Miku",), 2000),
            "other",
            max_tracks=2,
        )
        with pytest.raises(MusicPlaylistFullError):
            await playlist_repository.append(
                "shared-area",
                playlist.id,
                MusicTrack("netease", "third", "Third", (), None),
                "person",
                max_tracks=2,
            )
        summaries = await playlist_repository.list("shared-area")
        assert [(item.id, item.track_count) for item in summaries] == [(playlist.id, 2)]
        assert await playlist_repository.get("other-area", playlist.id) is None
        assert (
            await playlist_repository.remove(
                "shared-area",
                playlist.id,
                first_playlist_track.id,
            )
        ).removed is True
        compacted_playlist = await playlist_repository.get("shared-area", playlist.id)
        assert compacted_playlist is not None
        assert [(entry.id, entry.position) for entry in compacted_playlist.entries] == [
            (second_playlist_track.id, 1)
        ]
        imported_playlist = await playlist_repository.create_with_tracks(
            "shared-area",
            "网易云导入",
            "网易云导入".casefold(),
            (
                MusicTrack("youtube", "dQw4w9WgXcQ", "39 Live", ("初音未来",), 222000),
                MusicTrack(
                    "bilibili",
                    "BV1xx411c7mD:p=2",
                    "Tell Your World MV",
                    ("初音未来",),
                    245000,
                ),
            ),
            "person",
            max_tracks=3,
        )
        imported_readback = await playlist_repository.get(
            "shared-area",
            imported_playlist.id,
        )
        assert imported_readback is not None
        assert [entry.position for entry in imported_readback.entries] == [1, 2]
        assert [entry.track.source_id for entry in imported_readback.entries] == [
            "dQw4w9WgXcQ",
            "BV1xx411c7mD:p=2",
        ]
        assert [entry.track.source.value for entry in imported_readback.entries] == [
            "youtube",
            "bilibili",
        ]
        with pytest.raises(MusicPlaylistFullError):
            await playlist_repository.create_with_tracks(
                "shared-area",
                "过大导入",
                "过大导入".casefold(),
                tuple(entry.track for entry in imported_readback.entries),
                "person",
                max_tracks=1,
            )
        assert "过大导入" not in {
            summary.name for summary in await playlist_repository.list("shared-area")
        }

        renamed = await playlist_repository.rename(
            "shared-area",
            playlist.id,
            "Favorites",
            "favorites",
        )
        unchanged = await playlist_repository.rename(
            "shared-area",
            playlist.id,
            "Favorites",
            "favorites",
        )
        assert (renamed.old_name, renamed.new_name, renamed.changed) == (
            "夜间 电台",
            "Favorites",
            True,
        )
        assert unchanged.changed is False
        with pytest.raises(MusicPlaylistConflictError):
            await playlist_repository.rename(
                "shared-area",
                playlist.id,
                imported_playlist.name,
                imported_playlist.normalized_name,
            )
        with pytest.raises(MusicPlaylistNotFoundError):
            await playlist_repository.rename(
                "other-area",
                playlist.id,
                "Invisible",
                "invisible",
            )

        cleared = await playlist_repository.clear("shared-area", playlist.id)
        cleared_again = await playlist_repository.clear("shared-area", playlist.id)
        assert cleared.removed_track_count == 1
        assert cleared_again.removed_track_count == 0

        deleted = await playlist_repository.delete("shared-area", imported_playlist.id)
        deleted_again = await playlist_repository.delete("shared-area", imported_playlist.id)
        assert (deleted.deleted, deleted.removed_track_count) == (True, 2)
        assert deleted_again.deleted is False
        async with test_engine.connect() as connection:
            assert (
                await connection.scalar(
                    text("SELECT count(*) FROM music_playlist_tracks WHERE playlist_id = :id"),
                    {"id": imported_playlist.id},
                )
                == 0
            )

        append_clear = await playlist_repository.create(
            "shared-area",
            "append-clear-race",
            "append-clear-race",
            "person",
        )
        append_result, clear_result = await asyncio.gather(
            playlist_repository.append(
                "shared-area",
                append_clear.id,
                MusicTrack("netease", "race", "Race", (), None),
                "person",
                max_tracks=3,
            ),
            playlist_repository.clear("shared-area", append_clear.id),
        )
        assert append_result.playlist_id == append_clear.id
        assert clear_result.playlist_id == append_clear.id
        append_clear_readback = await playlist_repository.get("shared-area", append_clear.id)
        assert append_clear_readback is not None
        assert [entry.position for entry in append_clear_readback.entries] in ([], [1])

        append_delete = await playlist_repository.create(
            "shared-area",
            "append-delete-race",
            "append-delete-race",
            "person",
        )
        append_outcome, delete_outcome = await asyncio.gather(
            playlist_repository.append(
                "shared-area",
                append_delete.id,
                MusicTrack("netease", "late", "Late", (), None),
                "person",
                max_tracks=3,
            ),
            playlist_repository.delete("shared-area", append_delete.id),
            return_exceptions=True,
        )
        assert not isinstance(delete_outcome, BaseException)
        assert not isinstance(append_outcome, BaseException) or isinstance(
            append_outcome,
            MusicPlaylistNotFoundError,
        )
        assert await playlist_repository.get("shared-area", append_delete.id) is None
        async with test_engine.connect() as connection:
            assert (
                await connection.scalar(
                    text("SELECT count(*) FROM music_playlist_tracks WHERE playlist_id = :id"),
                    {"id": append_delete.id},
                )
                == 0
            )

        async with test_engine.begin() as connection:
            triggered_updated_at = await connection.scalar(
                text(
                    """
                    UPDATE channel_settings
                    SET chat_enabled = true
                    WHERE id = :id
                    RETURNING updated_at
                    """
                ),
                {"id": defaulted["id"]},
            )
            enum_columns = (
                (
                    await connection.execute(
                        text(
                            """
                        SELECT table_name, column_name, udt_name
                        FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND (table_name, column_name) IN (
                            ('agent_runs', 'status'),
                            ('agent_runs', 'stop_reason'),
                            ('agent_runs', 'selection_source'),
                            ('agent_tool_executions', 'effect'),
                            ('agent_tool_executions', 'status'),
                            ('agent_skill_resources', 'kind'),
                            ('agent_skills', 'ownership_kind'),
                            ('agent_skill_shares', 'status')
                          )
                        """
                        )
                    )
                )
                .mappings()
                .all()
            )
        assert triggered_updated_at > defaulted["updated_at"]
        assert {
            (row["table_name"], row["column_name"]): row["udt_name"] for row in enum_columns
        } == {
            ("agent_runs", "status"): "agent_run_status",
            ("agent_runs", "stop_reason"): "agent_stop_reason",
            ("agent_runs", "selection_source"): "model_selection_source",
            ("agent_tool_executions", "effect"): "tool_effect",
            ("agent_tool_executions", "status"): "tool_execution_status",
            ("agent_skill_resources", "kind"): "agent_skill_resource_kind",
            ("agent_skills", "ownership_kind"): "agent_skill_ownership_kind",
            ("agent_skill_shares", "status"): "agent_skill_share_status",
        }

        provider_id = uuid4()
        application_model_id = uuid4()
        preferred_model_id = uuid4()
        catalog_repository = SqlAlchemyProviderCatalogRepository(sessions)
        await catalog_repository.upsert_provider_bundle(
            LlmProvider(
                id=provider_id,
                alias="integration",
                display_name="Integration provider",
                protocol=ProviderProtocol.OPENAI_CHAT_COMPATIBLE,
                base_url="https://llm.example/v1",
                api_key="database-key",
                user_selectable=True,
                enabled=True,
            ),
            (
                LlmModel(
                    id=application_model_id,
                    provider_id=provider_id,
                    alias="application",
                    remote_model_name="application-model",
                    display_name="Application model",
                    enabled=True,
                    is_provider_default=True,
                    is_application_default=True,
                    capabilities=frozenset({ModelCapability.TOOL_CALLING}),
                ),
                LlmModel(
                    id=preferred_model_id,
                    provider_id=provider_id,
                    alias="preferred",
                    remote_model_name="preferred-model",
                    display_name="Preferred model",
                    enabled=True,
                    is_provider_default=False,
                    is_application_default=False,
                    capabilities=frozenset(
                        {
                            ModelCapability.TOOL_CALLING,
                            ModelCapability.STREAMING,
                        }
                    ),
                ),
            ),
        )
        async with test_engine.connect() as connection:
            provider_model_count = await connection.scalar(
                text(
                    """
                    SELECT count(*)
                    FROM llm_models
                    WHERE provider_id = :provider_id
                    """
                ),
                {"provider_id": provider_id},
            )
            model_indexes = {
                row["indexname"]: row["indexdef"]
                for row in (
                    await connection.execute(
                        text(
                            """
                            SELECT indexname, indexdef
                            FROM pg_indexes
                            WHERE schemaname = current_schema()
                              AND tablename = 'llm_models'
                            """
                        )
                    )
                )
                .mappings()
                .all()
            }
            provider_id_comment = await connection.scalar(
                text(
                    """
                    SELECT col_description('llm_models'::regclass, attnum)
                    FROM pg_attribute
                    WHERE attrelid = 'llm_models'::regclass
                      AND attname = 'provider_id'
                    """
                )
            )
        assert provider_model_count == 2
        assert "WHERE is_provider_default" in model_indexes["ux_llm_models_one_provider_default"]
        assert "multiple models may reference the same provider" in provider_id_comment
        async with sessions() as session:
            async with session.begin():
                session.add(
                    ChannelSettingsRecord(
                        area_id="area",
                        channel_id="channel",
                        chat_enabled=True,
                        default_model_id=application_model_id,
                    )
                )

        catalog = ReloadableProviderCatalog(catalog_repository)
        await catalog.reload()
        selection_repository = SqlAlchemyModelSelectionRepository(sessions)
        await selection_repository.set_user_model("person", preferred_model_id)
        selected = await ProviderSelectionService(
            catalog,
            selection_repository,
        ).resolve(
            ConversationKey("channel", "area", "channel", "person"),
            required_capabilities=frozenset({ModelCapability.TOOL_CALLING}),
        )

        assert selected.model.model_id == preferred_model_id
        assert selected.source is ModelSelectionSource.USER

        now = datetime.now(UTC)
        key = ConversationKey("channel", "area", "channel", "person")
        thread = AgentThread(
            id=uuid4(),
            key=key,
            selected_model_id=preferred_model_id,
            expires_at=now + timedelta(hours=1),
        )
        thread_repository = SqlAlchemyAgentThreadRepository(sessions)
        await thread_repository.add(thread)
        assert await thread_repository.get(key) == thread

        run_repository = SqlAlchemyAgentRunRepository(sessions)
        run_id = uuid4()
        running = AgentRunState(run_id).start(now)
        await run_repository.add(
            AgentRun(
                id=run_id,
                thread_id=thread.id,
                provider_id=provider_id,
                model_id=preferred_model_id,
                selection_source=ModelSelectionSource.USER,
                limits=AgentRunLimits(),
                state=running,
                heartbeat_at=now,
            )
        )
        heartbeat_at = now + timedelta(milliseconds=500)
        assert await run_repository.heartbeat(run_id, heartbeat_at) is True
        tool_execution_repository = SqlAlchemyToolExecutionRepository(sessions)
        tool_execution = ToolExecution(
            id=uuid4(),
            run_id=run_id,
            call_id="call-status",
            tool_name="get_agent_status",
            tool_version="1",
            effect=ToolEffect.READ,
            status=ToolExecutionStatus.STARTED,
            idempotency_key=f"{run_id}:call-status",
            input_payload={},
            output_payload=None,
            error_code="",
            started_at=now,
        )
        first_claim = await tool_execution_repository.claim(tool_execution)
        duplicate_claim = await tool_execution_repository.claim(tool_execution)
        semantic_duplicate = await tool_execution_repository.claim(
            replace(
                tool_execution,
                id=uuid4(),
                call_id="call-status-repeated",
            )
        )
        assert first_claim.created is True
        assert duplicate_claim.created is False
        assert semantic_duplicate.created is False
        assert semantic_duplicate.execution.call_id == "call-status"
        completed_tool = await tool_execution_repository.finish(
            run_id,
            "call-status",
            ToolExecutionStatus.SUCCEEDED,
            output={"mode": "agent"},
            error_code="",
        )
        assert completed_tool.output_payload == {"mode": "agent"}
        await run_repository.finish(
            running.finish(AgentStopReason.COMPLETED, now + timedelta(seconds=1)),
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        async with sessions() as session:
            completed = await session.scalar(
                select(AgentRunRecord).where(AgentRunRecord.id == run_id)
            )
        assert completed is not None
        assert completed.status == AgentRunStatus.SUCCEEDED
        assert completed.stop_reason == AgentStopReason.COMPLETED
        assert await run_repository.heartbeat(run_id, now + timedelta(seconds=2)) is False
        with pytest.raises(DBAPIError):
            async with test_engine.begin() as connection:
                await connection.execute(
                    text("UPDATE agent_runs SET status = 'unknown' WHERE id = :id"),
                    {"id": run_id},
                )

        message_repository = SqlAlchemyAgentMessageRepository(sessions)
        await message_repository.append(
            thread.id,
            run_id,
            (
                AgentMessage("user", "text", {"text": "question"}),
                AgentMessage(
                    "assistant",
                    "text",
                    {"text": "answer"},
                    input_tokens=10,
                    output_tokens=5,
                ),
            ),
        )
        assert await message_repository.count(thread.id) == 2
        loaded_messages = await message_repository.load(thread.id, limit=10)
        assert [message.content["text"] for message in loaded_messages] == [
            "question",
            "answer",
        ]
        assert [message.sequence for message in loaded_messages] == [1, 2]
        assert await thread_repository.save_summary(
            thread.id,
            "question and answer",
            2,
            expected_version=thread.version,
        )
        summarized_thread = await thread_repository.get(key)
        assert summarized_thread is not None
        assert summarized_thread.summary == "question and answer"
        assert summarized_thread.summary_through_sequence == 2
        assert summarized_thread.summary_version == 1
        assert (
            await message_repository.load(
                thread.id,
                limit=10,
                after_sequence=2,
            )
            == ()
        )

        memory_repository = SqlAlchemyMemoryRepository(sessions)
        memory_id = uuid4()
        memory_item = MemoryItem(
            id=memory_id,
            owner_person_id="person",
            namespace="explicit",
            content={"text": "likes jazz"},
            source_thread_id=thread.id,
            source_message_sequence=2,
            created_at=now,
            updated_at=now,
            last_used_at=None,
            expires_at=now + timedelta(days=30),
        )
        await memory_repository.add(memory_item)
        await memory_repository.set_preference("person", False)
        assert await memory_repository.preference("person") is False
        assert await memory_repository.count_active("person", now) == 1
        assert await memory_repository.count_active("other", now) == 0
        assert await memory_repository.list_active("person", now, limit=10) == (memory_item,)
        assert await memory_repository.delete("other", memory_id) is False
        await memory_repository.touch("person", (memory_id,), now)

        stale_id = uuid4()
        stale_at = now - timedelta(hours=2)
        await run_repository.add(
            AgentRun(
                id=stale_id,
                thread_id=thread.id,
                provider_id=provider_id,
                model_id=preferred_model_id,
                selection_source=ModelSelectionSource.USER,
                limits=AgentRunLimits(),
                state=AgentRunState(stale_id).start(stale_at),
                heartbeat_at=stale_at,
            )
        )
        await message_repository.append(
            thread.id,
            stale_id,
            (AgentMessage("user", "text", {"text": "stale question"}),),
        )
        assert (
            await run_repository.abandon_stale(
                now - timedelta(hours=1),
                now,
            )
            == 1
        )
        assert await message_repository.count(thread.id) == 2
        visible_after_abandon = await message_repository.load(thread.id, limit=10)
        assert [message.content["text"] for message in visible_after_abandon] == [
            "question",
            "answer",
        ]

        with pytest.raises(IntegrityError):
            async with sessions.begin() as session:
                session.add(
                    LlmModelRecord(
                        id=uuid4(),
                        provider_id=provider_id,
                        alias="duplicate-application-default",
                        remote_model_name="duplicate",
                        display_name="Duplicate",
                        enabled=True,
                        is_provider_default=False,
                        is_application_default=True,
                        capabilities=[],
                        limits={},
                        pricing={},
                    )
                )

        await thread_repository.delete(key)
        memory_after_thread_delete = await memory_repository.list_active(
            "person",
            now,
            limit=10,
        )
        assert len(memory_after_thread_delete) == 1
        assert memory_after_thread_delete[0].source_thread_id is None

        await _migrate(test_engine, "downgrade", "base")
    finally:
        if test_engine is not None:
            await test_engine.dispose()
        async with admin_engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin_engine.dispose()


async def _migrate(engine, operation: str, revision: str) -> None:
    async with engine.connect() as connection:
        await connection.run_sync(
            lambda sync_connection: _run_alembic(
                sync_connection,
                operation,
                revision,
            )
        )


def _run_alembic(connection, operation: str, revision: str) -> None:
    config = Config("alembic.ini")
    config.attributes["connection"] = connection
    getattr(command, operation)(config, revision)
