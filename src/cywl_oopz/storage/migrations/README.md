# Database migrations

Migrations are explicit deployment operations. After `DATABASE_URL` is available in the environment, run:

```bash
uv run alembic upgrade head
```

Create a revision after model changes with `uv run alembic revision --autogenerate -m "description"`, review it, then commit it. Never run migrations automatically from the bot process.
