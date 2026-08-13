"""Shared-playlist handlers and fresh reference resolution for music commands."""

from __future__ import annotations

from uuid import UUID

from cywl_oopz.commands.models import CommandRequest
from cywl_oopz.features.agent.models import AgentIdentity

from .command_handlers import MusicCommandView
from .command_parsing import (
    MusicArguments,
    PlaylistAddArguments,
    PlaylistCreateArguments,
    PlaylistImportArguments,
    PlaylistListArguments,
    PlaylistMutationAction,
    PlaylistMutationArguments,
    PlaylistPreviewArguments,
    PlaylistRemoveArguments,
    PlaylistRenameArguments,
    PlaylistShowArguments,
)
from .errors import MusicPlaylistNotFoundError
from .models import MusicPlaylist
from .playlists import MusicPlaylistService


class MusicPlaylistReferenceResolver:
    """Resolve request-local playlist indexes and entry positions through fresh reads."""

    def __init__(self, playlists: MusicPlaylistService) -> None:
        self._playlists = playlists

    async def playlist(self, identity: AgentIdentity, reference: str) -> MusicPlaylist:
        return await self._playlists.get(
            identity,
            await self.playlist_id(identity, reference),
        )

    async def playlist_id(self, identity: AgentIdentity, reference: str) -> UUID:
        try:
            return UUID(reference)
        except ValueError:
            pass
        normalized = reference.removeprefix("#")
        if not normalized.isdecimal() or int(normalized) < 1:
            raise MusicPlaylistNotFoundError("Playlist reference is invalid")
        playlists = await self._playlists.list(identity)
        index = int(normalized) - 1
        if index >= len(playlists):
            raise MusicPlaylistNotFoundError("Playlist index is outside the current area list")
        return playlists[index].id

    @staticmethod
    def entry_id(playlist: MusicPlaylist, reference: str) -> UUID:
        try:
            return UUID(reference)
        except ValueError:
            pass
        normalized = reference.removeprefix("#")
        if not normalized.isdecimal() or int(normalized) < 1:
            raise MusicPlaylistNotFoundError("Playlist entry reference is invalid")
        position = int(normalized)
        entry = next((item for item in playlist.entries if item.position == position), None)
        if entry is None:
            raise MusicPlaylistNotFoundError("Playlist entry position does not exist")
        return entry.id


class MusicPlaylistCommandHandler:
    _argument_types = (
        PlaylistListArguments,
        PlaylistShowArguments,
        PlaylistCreateArguments,
        PlaylistRenameArguments,
        PlaylistMutationArguments,
        PlaylistAddArguments,
        PlaylistRemoveArguments,
        PlaylistPreviewArguments,
        PlaylistImportArguments,
    )

    def __init__(
        self,
        playlists: MusicPlaylistService,
        view: MusicCommandView,
        references: MusicPlaylistReferenceResolver | None = None,
    ) -> None:
        self._playlists = playlists
        self._view = view
        self._references = references or MusicPlaylistReferenceResolver(playlists)

    def supports(self, arguments: MusicArguments) -> bool:
        return isinstance(arguments, self._argument_types)

    async def handle(
        self,
        request: CommandRequest,
        identity: AgentIdentity,
        arguments: MusicArguments,
    ) -> None:
        if isinstance(arguments, PlaylistListArguments):
            await request.responder.reply(
                self._view.playlists(await self._playlists.list(identity))
            )
            return
        if isinstance(arguments, PlaylistShowArguments):
            playlist = await self._references.playlist(identity, arguments.playlist)
            await request.responder.reply(self._view.playlist(playlist))
            return
        if isinstance(arguments, PlaylistCreateArguments):
            playlist = await self._playlists.create(identity, arguments.name)
            await request.responder.reply(f"✅ 已创建共享歌单 **{playlist.name}**。")
            return
        if isinstance(arguments, PlaylistRenameArguments):
            playlist = await self._references.playlist(identity, arguments.playlist)
            result = await self._playlists.rename(identity, playlist.id, arguments.name)
            message = (
                f"✅ **{result.old_name}** → **{result.new_name}**"
                if result.changed
                else f"共享歌单已经叫 **{result.new_name}**。"
            )
            await request.responder.reply(message)
            return
        if isinstance(arguments, PlaylistMutationArguments):
            await self._mutation(request, identity, arguments)
            return
        if isinstance(arguments, PlaylistAddArguments):
            playlist = await self._references.playlist(identity, arguments.playlist)
            entry = await self._playlists.add_input(
                identity,
                playlist.id,
                arguments.value,
                source=arguments.source,
            )
            await request.responder.reply(
                f"✅ 已把 {self._view.track(entry.track)} 加入 **{playlist.name}** · "
                f"第 {entry.position} 首"
            )
            return
        if isinstance(arguments, PlaylistRemoveArguments):
            playlist = await self._references.playlist(identity, arguments.playlist)
            result = await self._playlists.remove(
                identity,
                playlist.id,
                self._references.entry_id(playlist, arguments.entry),
            )
            await request.responder.reply(
                "✅ 已从共享歌单移除歌曲。" if result.removed else "歌单中没有这个条目。"
            )
            return
        if isinstance(arguments, PlaylistPreviewArguments):
            preview = await self._playlists.preview_netease(arguments.reference)
            await request.responder.reply(self._view.preview(preview))
            return
        assert isinstance(arguments, PlaylistImportArguments)
        result = await self._playlists.import_netease(
            identity,
            arguments.reference,
            name=arguments.name,
            allow_partial=arguments.allow_partial,
        )
        suffix = f" · 跳过 {result.skipped_track_count} 首" if result.skipped_track_count else ""
        await request.responder.reply(
            f"☁️ 已导入为 **{result.playlist.name}** · {result.imported_track_count} 首{suffix}"
        )

    async def _mutation(
        self,
        request: CommandRequest,
        identity: AgentIdentity,
        arguments: PlaylistMutationArguments,
    ) -> None:
        if arguments.action is PlaylistMutationAction.DELETE:
            playlist_id = await self._references.playlist_id(identity, arguments.playlist)
            result = await self._playlists.delete(identity, playlist_id)
            message = (
                f"🗑️ 已删除共享歌单 **{result.name}** · 移除 {result.removed_track_count} 首。"
                if result.deleted
                else "这个共享歌单已经不存在。"
            )
        elif arguments.action is PlaylistMutationAction.CLEAR:
            playlist = await self._references.playlist(identity, arguments.playlist)
            result = await self._playlists.clear(identity, playlist.id)
            message = (
                f"🧹 已清空共享歌单 **{result.name}** · 移除 {result.removed_track_count} 首。"
            )
        else:
            playlist = await self._references.playlist(identity, arguments.playlist)
            result = await self._playlists.load(identity, playlist.id)
            message = (
                f"▶ 已从 **{result.playlist_name}** 重建播放队列 · {result.queue.loaded_count} 首"
            )
        await request.responder.reply(message)
