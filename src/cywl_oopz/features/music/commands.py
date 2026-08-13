"""Typed command definition for music playback and shared playlists."""

from __future__ import annotations

from cywl_oopz.commands.catalog import CommandSpec
from cywl_oopz.commands.definitions import (
    CommandDefinition,
    CommandExecutionPolicy,
    ExecutionMode,
    PublicCommandAuthorization,
)
from cywl_oopz.commands.models import CommandRequest

from .command_handlers import (
    MusicCommandHandler,
)
from .command_parsing import MusicArguments, MusicArgumentsParser
from .command_playback import (
    MusicHelpCommandHandler,
    MusicModeCommandHandler,
    MusicPlaybackCommandHandler,
)
from .command_playlists import MusicPlaylistCommandHandler
from .command_rendering import MusicCommandRenderer
from .playlists import MusicPlaylistService
from .service import MusicRequestService

__all__ = ["MusicCommand", "MusicCommandRenderer"]


class MusicCommand:
    """Assemble explicit parser and subdomain handlers for the ``music`` group."""

    name = "music"
    description = "点歌并管理播放队列、播放模式和共享歌单。"
    category = "音乐"
    usage = (
        "music <关键词或 URL>",
        "music <play|pause|resume|skip|stop|leave|now|queue|clear>",
        "music mode <顺序|随机|单曲|列表|不循环>",
        "music playlist <list|show|create|rename|delete|add|remove|clear|load|import> ...",
    )
    examples = ("music 初音未来", "music mode 随机", "music playlist list")

    def __init__(
        self,
        music: MusicRequestService,
        playlists: MusicPlaylistService,
        command_prefix: str,
        renderer: MusicCommandRenderer | None = None,
    ) -> None:
        view = renderer or MusicCommandRenderer(command_prefix)
        self._parser = MusicArgumentsParser(view.usage())
        self._handler = MusicCommandHandler(
            (
                MusicHelpCommandHandler(view),
                MusicPlaybackCommandHandler(music, view),
                MusicModeCommandHandler(music, view),
                MusicPlaylistCommandHandler(playlists, view),
            )
        )

    def definition(self) -> CommandDefinition[MusicArguments]:
        return CommandDefinition(
            CommandSpec.from_command(self),
            self._parser,
            self,
            PublicCommandAuthorization(),
            CommandExecutionPolicy(ExecutionMode.BACKGROUND, timeout_seconds=120.0),
        )

    async def handle(self, request: CommandRequest, arguments: MusicArguments) -> None:
        await self._handler.handle(request, arguments)
