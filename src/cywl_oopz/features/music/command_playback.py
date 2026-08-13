"""Playback, search, queue and mode handlers for direct music commands."""

from __future__ import annotations

from cywl_oopz.commands.models import CommandRequest
from cywl_oopz.features.agent.models import AgentIdentity

from .command_handlers import MusicCommandView
from .command_parsing import (
    MusicArguments,
    MusicControlAction,
    MusicControlArguments,
    MusicHelpArguments,
    MusicModeArguments,
    MusicPlayArguments,
    MusicQueueArguments,
    MusicSearchArguments,
    MusicSourcesArguments,
)
from .service import MusicRequestService


class MusicHelpCommandHandler:
    def __init__(self, view: MusicCommandView) -> None:
        self._view = view

    def supports(self, arguments: MusicArguments) -> bool:
        return isinstance(arguments, MusicHelpArguments)

    async def handle(
        self,
        request: CommandRequest,
        identity: AgentIdentity,
        arguments: MusicArguments,
    ) -> None:
        del identity, arguments
        await request.responder.reply(self._view.usage())


class MusicPlaybackCommandHandler:
    _argument_types = (
        MusicPlayArguments,
        MusicSearchArguments,
        MusicSourcesArguments,
        MusicQueueArguments,
        MusicControlArguments,
    )

    def __init__(self, music: MusicRequestService, view: MusicCommandView) -> None:
        self._music = music
        self._view = view

    def supports(self, arguments: MusicArguments) -> bool:
        return isinstance(arguments, self._argument_types)

    async def handle(
        self,
        request: CommandRequest,
        identity: AgentIdentity,
        arguments: MusicArguments,
    ) -> None:
        if isinstance(arguments, MusicPlayArguments):
            result = await self._music.enqueue_input(
                identity,
                arguments.value,
                source=arguments.source,
                idempotency_key=(
                    f"music-command:{identity.source_message_id}"
                    if identity.source_message_id
                    else ""
                ),
            )
            await request.responder.reply(
                f"🎵 已加入 {self._view.track(result.item.track)} · 队列第 {result.position} 位"
            )
            return
        if isinstance(arguments, MusicSearchArguments):
            tracks = await self._music.search(
                arguments.query,
                source=arguments.source,
                limit=5,
            )
            await request.responder.reply(self._view.tracks(tracks))
            return
        if isinstance(arguments, MusicSourcesArguments):
            await request.responder.reply(
                self._view.sources(
                    await self._music.health(),
                    default_source=self._music.default_source,
                )
            )
            return
        if isinstance(arguments, MusicQueueArguments):
            await request.responder.reply(self._view.queue(await self._music.queue(identity)))
            return
        assert isinstance(arguments, MusicControlArguments)
        if arguments.action is MusicControlAction.SKIP:
            applied = await self._music.skip(identity)
            message = "⏭️ 已切换下一首。" if applied else "当前没有正在播放的歌曲。"
        elif arguments.action is MusicControlAction.PAUSE:
            applied = await self._music.pause(identity)
            message = "⏸️ 已暂停播放。" if applied else "当前没有可暂停的歌曲。"
        elif arguments.action is MusicControlAction.RESUME:
            applied = await self._music.resume(identity)
            message = "▶ 已继续播放。" if applied else "当前没有已暂停的歌曲。"
        else:
            result = await self._music.clear(identity)
            stopped = "已停止当前歌曲，" if result.stopped_current else ""
            message = f"🧹 {stopped}已从播放队列移除 {result.removed_count} 首。"
        await request.responder.reply(message)


class MusicModeCommandHandler:
    def __init__(self, music: MusicRequestService, view: MusicCommandView) -> None:
        self._music = music
        self._view = view

    def supports(self, arguments: MusicArguments) -> bool:
        return isinstance(arguments, MusicModeArguments)

    async def handle(
        self,
        request: CommandRequest,
        identity: AgentIdentity,
        arguments: MusicArguments,
    ) -> None:
        assert isinstance(arguments, MusicModeArguments)
        if arguments.query_only:
            snapshot = await self._music.queue(identity)
            message = self._view.policy(
                snapshot.policy.order,
                snapshot.policy.repeat,
                changed=False,
            )
        else:
            result = await self._music.set_policy(
                identity,
                order=arguments.order,
                repeat=arguments.repeat,
            )
            message = self._view.policy(
                result.policy.order,
                result.policy.repeat,
                changed=result.changed,
            )
        await request.responder.reply(message)
