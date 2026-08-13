"""Coordination contracts and root handler for typed music commands."""

from __future__ import annotations

import logging
from typing import Protocol

from cywl_oopz.commands.models import CommandRequest, CommandScope
from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.agent.models import AgentIdentity
from cywl_oopz.features.chat.models import ConversationKey

from .command_errors import MusicCommandErrorPresenter
from .command_parsing import MusicArguments
from .errors import MusicError
from .models import (
    MusicPlaylist,
    MusicPlaylistSummary,
    MusicProviderHealth,
    MusicQueueSnapshot,
    MusicSourceKind,
    MusicTrack,
    NeteasePlaylistSnapshot,
    PlaybackOrder,
    RepeatPolicy,
)

logger = logging.getLogger(__name__)


class MusicCommandView(Protocol):
    """Rendering operations used by music handlers."""

    def usage(self) -> str: ...

    def tracks(self, tracks: tuple[MusicTrack, ...]) -> str: ...

    def sources(
        self,
        health: tuple[MusicProviderHealth, ...],
        *,
        default_source: MusicSourceKind,
    ) -> str: ...

    def queue(self, snapshot: MusicQueueSnapshot) -> str: ...

    def policy(self, order: PlaybackOrder, repeat: RepeatPolicy, *, changed: bool) -> str: ...

    def playlists(self, playlists: tuple[MusicPlaylistSummary, ...]) -> str: ...

    def playlist(self, playlist: MusicPlaylist) -> str: ...

    def preview(self, playlist: NeteasePlaylistSnapshot) -> str: ...

    def track(self, track: MusicTrack) -> str: ...


class MusicSubcommandHandler(Protocol):
    """One independently testable part of the music command namespace."""

    def supports(self, arguments: MusicArguments) -> bool: ...

    async def handle(
        self,
        request: CommandRequest,
        identity: AgentIdentity,
        arguments: MusicArguments,
    ) -> None: ...


class MusicCommandHandler:
    """Coordinate typed subcommands and one shared expected-error presenter."""

    def __init__(
        self,
        handlers: tuple[MusicSubcommandHandler, ...],
        errors: MusicCommandErrorPresenter | None = None,
    ) -> None:
        if not handlers:
            raise ValueError("Music command requires at least one subcommand handler")
        self._handlers = handlers
        self._errors = errors or MusicCommandErrorPresenter()

    async def handle(self, request: CommandRequest, arguments: MusicArguments) -> None:
        identity = self._identity(request)
        handler = next((item for item in self._handlers if item.supports(arguments)), None)
        if handler is None:
            raise RuntimeError(f"No music handler registered for {type(arguments).__name__}")
        try:
            await handler.handle(request, identity, arguments)
        except (MusicError, DatabaseError) as exc:
            logger.info(
                "Music command rejected: conversation=%s error=%s",
                self._conversation_ref(identity.conversation),
                exception_kind(exc),
            )
            await request.responder.reply(self._errors.message(exc))

    @staticmethod
    def _identity(request: CommandRequest) -> AgentIdentity:
        private = request.location.scope is CommandScope.PRIVATE
        conversation = ConversationKey(
            "private" if private else "channel",
            "" if private else request.location.area_id,
            "" if private else request.location.channel_id,
            request.actor.person_id,
        )
        return AgentIdentity(
            request.actor.person_id,
            conversation,
            source_message_id=request.source.message_id,
            transport_channel_id=request.location.channel_id,
        )

    @staticmethod
    def _conversation_ref(conversation: ConversationKey) -> str:
        return opaque_ref(
            conversation.scope,
            conversation.area_id,
            conversation.channel_id,
            conversation.person_id,
        )
