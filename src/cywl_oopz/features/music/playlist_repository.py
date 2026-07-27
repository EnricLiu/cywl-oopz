"""PostgreSQL repository for area-shared music playlists."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.storage.models import MusicPlaylistRecord, MusicPlaylistTrackRecord

from .errors import (
    MusicPlaylistConflictError,
    MusicPlaylistFullError,
    MusicPlaylistNotFoundError,
)
from .models import (
    MusicPlaylist,
    MusicPlaylistEntry,
    MusicPlaylistSummary,
    MusicTrack,
    PlaylistTrackRemoval,
)

logger = logging.getLogger(__name__)


class SqlAlchemyMusicPlaylistRepository:
    """Use short transactions and lock only the playlist being mutated."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(
        self,
        area_id: str,
        name: str,
        normalized_name: str,
        created_by_person_id: str,
    ) -> MusicPlaylist:
        now = datetime.now(UTC)
        playlist_id = uuid4()
        try:
            async with self._sessions() as session:
                async with session.begin():
                    session.add(
                        MusicPlaylistRecord(
                            id=playlist_id,
                            area_id=area_id,
                            name=name,
                            normalized_name=normalized_name,
                            created_by_person_id=created_by_person_id,
                            created_at=now,
                            updated_at=now,
                        )
                    )
        except IntegrityError as exc:
            raise MusicPlaylistConflictError(
                "This area already has a playlist with that name"
            ) from exc
        except SQLAlchemyError as exc:
            raise _database_error("create music playlist", exc) from exc
        return MusicPlaylist(
            playlist_id,
            area_id,
            name,
            normalized_name,
            created_by_person_id,
            (),
            now,
            now,
        )

    async def list(self, area_id: str) -> tuple[MusicPlaylistSummary, ...]:
        track_count = (
            select(func.count(MusicPlaylistTrackRecord.id))
            .where(MusicPlaylistTrackRecord.playlist_id == MusicPlaylistRecord.id)
            .correlate(MusicPlaylistRecord)
            .scalar_subquery()
        )
        try:
            async with self._sessions() as session:
                rows = (
                    await session.execute(
                        select(MusicPlaylistRecord, track_count.label("track_count"))
                        .where(MusicPlaylistRecord.area_id == area_id)
                        .order_by(
                            MusicPlaylistRecord.updated_at.desc(),
                            MusicPlaylistRecord.name,
                        )
                    )
                ).all()
        except SQLAlchemyError as exc:
            raise _database_error("list music playlists", exc) from exc
        return tuple(
            MusicPlaylistSummary(
                record.id,
                record.area_id,
                record.name,
                int(count),
                record.updated_at,
            )
            for record, count in rows
        )

    async def get(self, area_id: str, playlist_id: UUID) -> MusicPlaylist | None:
        try:
            async with self._sessions() as session:
                record = await session.scalar(
                    select(MusicPlaylistRecord).where(
                        MusicPlaylistRecord.id == playlist_id,
                        MusicPlaylistRecord.area_id == area_id,
                    )
                )
                if record is None:
                    return None
                tracks = (
                    await session.scalars(
                        select(MusicPlaylistTrackRecord)
                        .where(MusicPlaylistTrackRecord.playlist_id == playlist_id)
                        .order_by(MusicPlaylistTrackRecord.position)
                    )
                ).all()
        except SQLAlchemyError as exc:
            raise _database_error("load music playlist", exc) from exc
        return self._to_playlist(record, tracks)

    async def append(
        self,
        area_id: str,
        playlist_id: UUID,
        track: MusicTrack,
        added_by_person_id: str,
        *,
        max_tracks: int,
    ) -> MusicPlaylistEntry:
        now = datetime.now(UTC)
        entry_id = uuid4()
        try:
            async with self._sessions() as session:
                async with session.begin():
                    playlist = await self._locked_playlist(session, area_id, playlist_id)
                    count = (
                        await session.scalar(
                            select(func.count(MusicPlaylistTrackRecord.id)).where(
                                MusicPlaylistTrackRecord.playlist_id == playlist_id
                            )
                        )
                        or 0
                    )
                    if count >= max_tracks:
                        raise MusicPlaylistFullError("Music playlist is full")
                    position = count + 1
                    session.add(
                        MusicPlaylistTrackRecord(
                            id=entry_id,
                            playlist_id=playlist_id,
                            position=position,
                            source=track.source,
                            source_id=track.source_id,
                            title=track.title,
                            artists=list(track.artists),
                            duration_ms=track.duration_ms,
                            added_by_person_id=added_by_person_id,
                            created_at=now,
                        )
                    )
                    playlist.updated_at = now
        except MusicPlaylistNotFoundError:
            raise
        except MusicPlaylistFullError:
            raise
        except SQLAlchemyError as exc:
            raise _database_error("append music playlist track", exc) from exc
        return MusicPlaylistEntry(
            entry_id,
            playlist_id,
            position,
            track,
            added_by_person_id,
            now,
        )

    async def remove(
        self,
        area_id: str,
        playlist_id: UUID,
        entry_id: UUID,
    ) -> PlaylistTrackRemoval:
        try:
            async with self._sessions() as session:
                async with session.begin():
                    playlist = await self._locked_playlist(session, area_id, playlist_id)
                    position = await session.scalar(
                        select(MusicPlaylistTrackRecord.position).where(
                            MusicPlaylistTrackRecord.id == entry_id,
                            MusicPlaylistTrackRecord.playlist_id == playlist_id,
                        )
                    )
                    if position is None:
                        return PlaylistTrackRemoval(playlist_id, entry_id, False)
                    await session.execute(
                        delete(MusicPlaylistTrackRecord).where(
                            MusicPlaylistTrackRecord.id == entry_id,
                            MusicPlaylistTrackRecord.playlist_id == playlist_id,
                        )
                    )
                    await session.execute(
                        update(MusicPlaylistTrackRecord)
                        .where(
                            MusicPlaylistTrackRecord.playlist_id == playlist_id,
                            MusicPlaylistTrackRecord.position > position,
                        )
                        .values(position=MusicPlaylistTrackRecord.position - 1)
                    )
                    playlist.updated_at = datetime.now(UTC)
        except MusicPlaylistNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise _database_error("remove music playlist track", exc) from exc
        return PlaylistTrackRemoval(playlist_id, entry_id, True)

    @staticmethod
    async def _locked_playlist(
        session: AsyncSession,
        area_id: str,
        playlist_id: UUID,
    ) -> MusicPlaylistRecord:
        playlist = await session.scalar(
            select(MusicPlaylistRecord)
            .where(
                MusicPlaylistRecord.id == playlist_id,
                MusicPlaylistRecord.area_id == area_id,
            )
            .with_for_update()
        )
        if playlist is None:
            raise MusicPlaylistNotFoundError("Music playlist was not found in this area")
        return playlist

    @classmethod
    def _to_playlist(
        cls,
        record: MusicPlaylistRecord,
        tracks: list[MusicPlaylistTrackRecord],
    ) -> MusicPlaylist:
        return MusicPlaylist(
            record.id,
            record.area_id,
            record.name,
            record.normalized_name,
            record.created_by_person_id,
            tuple(cls._to_entry(track) for track in tracks),
            record.created_at,
            record.updated_at,
        )

    @staticmethod
    def _to_entry(record: MusicPlaylistTrackRecord) -> MusicPlaylistEntry:
        artists = record.artists if isinstance(record.artists, list) else []
        return MusicPlaylistEntry(
            record.id,
            record.playlist_id,
            record.position,
            MusicTrack(
                record.source,
                record.source_id,
                record.title,
                tuple(str(artist) for artist in artists),
                record.duration_ms,
            ),
            record.added_by_person_id,
            record.created_at,
        )


def _database_error(operation: str, error: SQLAlchemyError) -> DatabaseError:
    logger.warning(
        "Music playlist persistence failed: operation=%s error=%s",
        operation,
        type(error).__name__,
    )
    return DatabaseError(f"Failed to {operation}")
