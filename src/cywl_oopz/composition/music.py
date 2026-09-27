"""Music sources, playback, playlists, and their Agent adapters."""

from __future__ import annotations

from dataclasses import dataclass

from oopz_sdk import OopzBot
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.features.agent.tools.music import (
    ClearMusicQueueTool,
    EnqueueMusicTool,
    GetMusicQueueTool,
    PauseMusicTool,
    ResumeMusicTool,
    SearchMusicCatalogTool,
    SetMusicPlaybackModeTool,
    SkipMusicTool,
)
from cywl_oopz.features.agent.tools.playlists import (
    AddMusicPlaylistTrackTool,
    ClearMusicPlaylistTool,
    CreateMusicPlaylistTool,
    DeleteMusicPlaylistTool,
    GetMusicPlaylistTool,
    ImportNeteasePlaylistTool,
    ListMusicPlaylistsTool,
    LoadMusicPlaylistTool,
    PreviewNeteasePlaylistTool,
    RemoveMusicPlaylistTrackTool,
    RenameMusicPlaylistTool,
)
from cywl_oopz.features.agent.tools.ports import AgentTool
from cywl_oopz.features.music.bilibili import BilibiliMusicProvider
from cywl_oopz.features.music.catalog import CompositeMusicCatalog, MusicProviderRegistry
from cywl_oopz.features.music.models import MusicSourceKind
from cywl_oopz.features.music.netease import NeteaseMusicProvider
from cywl_oopz.features.music.playlist_repository import SqlAlchemyMusicPlaylistRepository
from cywl_oopz.features.music.playlists import MusicPlaylistService
from cywl_oopz.features.music.service import MusicRequestService
from cywl_oopz.features.music.youtube import YouTubeMusicProvider
from cywl_oopz.integrations.media.ytdlp_runner import YtDlpProcessRunner
from cywl_oopz.integrations.oopz.music import OopzMusicVoiceGateway
from cywl_oopz.integrations.oopz.voice_channel_session import OopzVoiceChannelSessionManager
from cywl_oopz.settings import (
    AppSettings,
)


@dataclass(slots=True)
class MusicComponents:
    music: MusicRequestService | None = None
    music_catalog: CompositeMusicCatalog | None = None
    netease_music_provider: NeteaseMusicProvider | None = None
    bilibili_music_provider: BilibiliMusicProvider | None = None
    youtube_music_provider: YouTubeMusicProvider | None = None
    music_playlists: MusicPlaylistService | None = None
    music_voice: OopzMusicVoiceGateway | None = None
    music_ytdlp_runner: YtDlpProcessRunner | None = None
    tools: tuple[AgentTool, ...] = ()


def build_music(
    settings: AppSettings,
    bot: OopzBot,
    voice_channels: OopzVoiceChannelSessionManager,
    sessions: async_sessionmaker[AsyncSession],
) -> MusicComponents:
    """Build enabled music sources and their tools against the shared voice backend."""
    components = MusicComponents()
    agent_tools: list[AgentTool] = []
    if settings.music.enabled:
        if any(
            source in {MusicSourceKind.YOUTUBE, MusicSourceKind.BILIBILI}
            for source in settings.music.enabled_sources
        ):
            components.music_ytdlp_runner = YtDlpProcessRunner(settings.music_ytdlp)
        music_providers = []
        for source in settings.music.enabled_sources:
            if source is MusicSourceKind.NETEASE:
                components.netease_music_provider = NeteaseMusicProvider(settings.music)
                music_providers.append(components.netease_music_provider)
            elif source is MusicSourceKind.BILIBILI:
                assert components.music_ytdlp_runner is not None
                components.bilibili_music_provider = BilibiliMusicProvider(
                    settings.music,
                    settings.music_ytdlp,
                    components.music_ytdlp_runner,
                )
                music_providers.append(components.bilibili_music_provider)
            elif source is MusicSourceKind.YOUTUBE:
                assert components.music_ytdlp_runner is not None
                components.youtube_music_provider = YouTubeMusicProvider(
                    settings.music,
                    settings.music_ytdlp,
                    components.music_ytdlp_runner,
                )
                music_providers.append(components.youtube_music_provider)
        components.music_catalog = CompositeMusicCatalog(
            MusicProviderRegistry(music_providers),
            settings.music.default_source,
        )
        components.music_voice = OopzMusicVoiceGateway(
            bot,
            voice_channels,
            settings.audio,
        )
        components.music = MusicRequestService(
            settings.music,
            components.music_catalog,
            components.music_voice,
        )
        components.music_playlists = MusicPlaylistService(
            settings.music,
            SqlAlchemyMusicPlaylistRepository(sessions),
            components.music,
            components.netease_music_provider,
        )
        music_tool_options = {
            "timeout_seconds": settings.agent.tool_timeout_seconds,
            "max_output_characters": settings.agent.max_tool_result_characters,
        }
        music_import_tool_options = {
            **music_tool_options,
            "timeout_seconds": max(
                settings.agent.tool_timeout_seconds,
                settings.music.request_timeout_seconds * 2 + 2,
            ),
        }
        agent_tools.extend(
            (
                SearchMusicCatalogTool(components.music, **music_tool_options),
                EnqueueMusicTool(components.music, **music_tool_options),
                GetMusicQueueTool(components.music, **music_tool_options),
                SkipMusicTool(components.music, **music_tool_options),
                PauseMusicTool(components.music, **music_tool_options),
                ResumeMusicTool(components.music, **music_tool_options),
                ClearMusicQueueTool(components.music, **music_tool_options),
                SetMusicPlaybackModeTool(components.music, **music_tool_options),
                CreateMusicPlaylistTool(components.music_playlists, **music_tool_options),
                ListMusicPlaylistsTool(components.music_playlists, **music_tool_options),
                GetMusicPlaylistTool(components.music_playlists, **music_tool_options),
                AddMusicPlaylistTrackTool(components.music_playlists, **music_tool_options),
                RemoveMusicPlaylistTrackTool(
                    components.music_playlists,
                    **music_tool_options,
                ),
                RenameMusicPlaylistTool(components.music_playlists, **music_tool_options),
                DeleteMusicPlaylistTool(components.music_playlists, **music_tool_options),
                ClearMusicPlaylistTool(components.music_playlists, **music_tool_options),
                LoadMusicPlaylistTool(components.music_playlists, **music_tool_options),
                PreviewNeteasePlaylistTool(
                    components.music_playlists,
                    **music_import_tool_options,
                ),
                ImportNeteasePlaylistTool(
                    components.music_playlists,
                    **music_import_tool_options,
                ),
            )
        )
    components.tools = tuple(agent_tools)
    return components
