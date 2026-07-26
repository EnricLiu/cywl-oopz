"""Alembic environment for asynchronous PostgreSQL migrations."""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from cywl_oopz.storage.models import Base
from cywl_oopz.storage.url import normalize_asyncpg_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

load_dotenv(find_dotenv(usecwd=True), override=False)
target_metadata = Base.metadata


def database_url() -> str:
    """Read the deployment-local database URL without logging it."""
    return normalize_asyncpg_url(os.getenv("DATABASE_URL", ""))


def run_migrations_offline() -> None:
    """Generate SQL without opening a database connection."""
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure Alembic inside the synchronous SQLAlchemy bridge."""
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an asyncpg connection when the caller did not supply one."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = database_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    """Use an injected sync bridge for tests, or own an asyncpg connection."""
    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        do_run_migrations(supplied_connection)
        return
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
