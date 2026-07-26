"""Asynchronous PostgreSQL engine lifecycle management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from cywl_oopz.core.errors import DatabaseError
from cywl_oopz.settings import DatabaseSettings


class Database:
    """Owns one PostgreSQL connection pool for the whole bot process."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._engine: AsyncEngine = create_async_engine(
            settings.url,
            pool_pre_ping=True,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout_seconds,
            pool_recycle=settings.pool_recycle_seconds,
        )
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Expose a factory to repositories without exposing the engine."""
        return self._sessions

    async def start(self) -> None:
        """Verify the pool can execute a minimal, read-only health query."""
        try:
            async with self._engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise DatabaseError("PostgreSQL connection check failed") from exc

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Provide a transaction that commits only after a successful caller block."""
        async with self._sessions() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    async def close(self) -> None:
        """Release all pool connections during shutdown."""
        await self._engine.dispose()
