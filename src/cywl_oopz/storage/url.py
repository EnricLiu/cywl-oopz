"""Database URL helpers shared by runtime configuration and Alembic."""

from __future__ import annotations

from cywl_oopz.core.errors import ConfigurationError


def normalize_asyncpg_url(database_url: str) -> str:
    """Accept common PostgreSQL URLs and return a SQLAlchemy asyncpg URL."""
    value = database_url.strip()
    if not value:
        raise ConfigurationError("DATABASE_URL environment variable is required")
    if value.startswith("postgres://"):
        value = "postgresql://" + value.removeprefix("postgres://")
    if value.startswith("postgresql://"):
        value = "postgresql+asyncpg://" + value.removeprefix("postgresql://")
    if not value.startswith("postgresql+asyncpg://"):
        raise ConfigurationError("DATABASE_URL must use a PostgreSQL connection URL")
    return value
