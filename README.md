# CYWL OOPZ Bot

An asynchronous [OOPZ](https://oopz.com) community entertainment bot built on `oopz-sdk`.
The bot plays the role of CYWL (虚拟歌手初音未来 / Hatsune Miku) and supports AI Agent
chat, music playback, web search, browser-backed reading, and experimental realtime
voice conversation.

## Features

- **OOPZ lifecycle** — WebSocket connection, automatic domain/area subscription, reconnect.
- **Built-in commands** — `/ping`, `/help`, `/status`.
- **AI Agent chat** — Pydantic AI–based agent loop; provider and model are stored in
  PostgreSQL and can be switched per-channel. Per-conversation history, summarization,
  memory, tool calls, concurrency budgets, timeout, and cancellation.
- **Single-reply live display** — the agent creates one OOPZ reply and edits it in-place
  to show thinking, tool steps, and final content; output is capped at 2 000 UTF-16 units.
- **Persistent skills** — user- and admin-owned skill definitions stored in PostgreSQL;
  the agent can load and apply them at conversation time.
- **Music playback** — NeteaseCloudMusicApi-compatible catalog search, per-voice-channel
  queue, automatic progression, and playlist import.
- **Web search** — DuckDuckGo public search, no API key required.
- **Browser tools** — `agent-browser` MCP subprocess; per-conversation isolated sessions
  supporting page reading, snapshots, click, fill, and key press. Disabled by default;
  interaction additionally requires per-channel allow-listing.
- **Experimental voice conversation** — realtime STT/LLM/TTS pipeline backed by Qwen
  Omni/Audio; per-voice-channel sessions with idle timeout, barge-in, and delegated
  background tasks. Disabled by default (`CYWL_VOICE_ENABLED=false`).
- **PostgreSQL persistence** — SQLAlchemy async engine, `asyncpg`, Alembic migrations for
  chat history, agent threads, memories, channel settings, skill library, and voice state.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- PostgreSQL 14+ (or compatible)
- *(Optional)* `agent-browser` 0.33.x for browser tools
  ```bash
  npm install -g agent-browser@0.33.0
  agent-browser install
  ```

## Installation

```bash
git clone --recurse-submodules https://github.com/EnricLiu/cywl-oopz.git
cd cywl-oopz
uv sync --all-groups
```

> **Note:** The `sdk/` submodule (`oopz-sdk`) is required. If you cloned without
> `--recurse-submodules`, run `git submodule update --init`.

## Configuration

Copy the example file and fill in the required values:

```bash
cp .env.example .env
```

The `.env` file is git-ignored and must never be committed. Required variables:

| Variable | Description |
|---|---|
| `OOPZ_DEVICE_ID` | Device identifier from the OOPZ SDK login flow |
| `OOPZ_PERSON_UID` | Bot account person UID |
| `OOPZ_JWT_TOKEN` | Authentication token |
| `DATABASE_URL` | PostgreSQL connection string, e.g. `******host:5432/dbname` |

All other variables have documented defaults in `.env.example`. Key opt-in features:

| Variable | Default | Notes |
|---|---|---|
| `CYWL_CHAT_ENABLED` | `false` | Legacy direct-LLM chat; requires `CYWL_LLM_*` |
| `CYWL_AGENT_MODE` | `legacy` | Set to `agent` to use the PostgreSQL-backed provider catalog |
| `CYWL_MUSIC_ENABLED` | `false` | Requires a NeteaseCloudMusicApi-compatible endpoint |
| `CYWL_WEB_BROWSER_ENABLED` | `false` | Requires `agent-browser` on PATH |
| `CYWL_VOICE_ENABLED` | `false` | Experimental; provider config stored in PostgreSQL |

### Agent mode provider setup

In `agent` mode, LLM provider and model configuration is stored in PostgreSQL rather than
in environment variables. Insert a row into the `llm_providers` table with your API base
URL, key, and available models before starting the bot.

## Database migration

Run migrations before the first start (and after any upgrade):

```bash
uv run alembic upgrade head
```

The application performs a read-only `SELECT 1` health check on startup but never applies
migrations automatically.

## Running

```bash
uv run cywl-oopz
```

Or from the repository root:

```bash
python main.py
```

## Commands

| Command | Description |
|---|---|
| `/ping` | Check whether the bot is online. |
| `/help` | List available commands. |
| `/status` | Show component health (OOPZ, database, LLM). |
| `/chat <text>` | Start or continue your text conversation (legacy mode). |
| Mention bot + text | Start or continue your text conversation. |
| Direct message | Chat in a private session. |
| `/new` | Clear your conversation history in the current scope. |
| `/cancel` | Cancel the in-progress reply and release resources. |
| `/model [name]` | View or switch the current LLM model (allow-listed users only). |
| `/chat-status` | Show session stats, model, history size, and cooldown. |
| `/voice start` | Start an experimental realtime voice session in the current voice channel. |
| `/voice stop` | Stop the active voice session. |
| `/voice status` | Show voice session state and current model. |
| `/voice models` | List selectable voice models. |
| `/voice model <name>` | Switch voice model for the current session. |
| `/voice voice <name>` | Select a TTS voice variant. |

> Voice commands require `CYWL_VOICE_ENABLED=true` and a configured voice provider in
> PostgreSQL. They are experimental.

## Project layout

```
src/cywl_oopz/
  application.py            Composition root and resource lifecycle
  core/                     Shared utilities: config errors, health, tasks
  commands/                 Command protocol, router, and built-in commands
  features/agent/           Agent loop, provider/model catalog, tools, memory, skills
  features/chat/            Conversation commands, legacy LLM chat, streaming progress
  features/music/           Catalog search, per-channel queue, and playback state
  features/voice/           Realtime voice session, audio pipeline, task delegation
  features/web/             Search and browser domain services
  integrations/oopz/        OOPZ SDK adapters (presenter, editable messages, music, voice)
  integrations/voice/       Qwen Omni/Audio provider and protocol adapters
  integrations/web/         DuckDuckGo and agent-browser MCP adapters
  storage/                  SQLAlchemy models, repositories, Alembic migrations
sdk/                        oopz-sdk submodule (integration boundary)
tests/                      Unit and contract tests; no real credentials required
```

## Development

Format, lint, and test:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

Run the CI quality checks locally (mirrors `.github/workflows/quality.yml`):

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

Live integration tests (Qwen Audio, OOPZ media) are opt-in via environment variables
documented in `.env.example` and are skipped by default.

## Security

- Never commit real credentials. All runtime secrets (OOPZ tokens, LLM API keys,
  database passwords) belong in your local `.env` file or PostgreSQL, both of which are
  git-ignored.
- `.env.example` contains only placeholders and safe defaults; inspect it before copying.
- To report a security issue, open a private security advisory on GitHub or contact the
  maintainer directly.


## License

[MIT](LICENSE)
