# CYWL OOPZ Bot

[简体中文](README.zh-CN.md)

CYWL is an asynchronous community entertainment bot for
[OOPZ](https://oopz.com). It presents as CYWL—Hatsune Miku—and combines
conversation, agent tools, music, web research, and experimental realtime voice
conversation in one bot process.

The project is built around the OOPZ SDK, Python async I/O, PostgreSQL, and a
small set of replaceable feature integrations. `BotApplication` is the
composition root that owns the OOPZ client and the lifecycle of every service.

## What it can do

- **Connect to OOPZ reliably** — manages the WebSocket client, subscriptions,
  reconnects, and health reporting.
- **Chat in two modes** — use a direct OpenAI-compatible LLM route, or the
  PostgreSQL-backed Pydantic AI Agent route. Both preserve scoped conversation
  history and support cancellation, timeouts, and bounded concurrency.
- **Run a stateful Agent** — select providers and models per conversation,
  summarize long threads, retain opt-in user memory, load persistent skills,
  and call only policy-authorized tools. Agent progress can optionally update a
  single OOPZ reply in place.
- **Play music in voice channels** — search a
  NeteaseCloudMusicApi-compatible catalog, maintain a serialized queue per
  voice channel, control playback, and manage/import playlists.
- **Research the web** — search DuckDuckGo without an API key and read pages
  with an optional, isolated `agent-browser` MCP session. Browser interaction
  is separately opt-in and channel-controlled.
- **Talk in voice channels (experimental)** — run Qwen Omni/Audio-backed
  realtime STT/LLM/TTS conversations with per-channel session ownership,
  interruption handling, idle limits, and delegated Agent tasks.
- **Persist runtime state** — use SQLAlchemy and Alembic for conversations,
  Agent runs, provider/model catalogs, memories, skills, channel policies,
  playlists, and voice sessions.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL 14+ or a compatible server
- OOPZ account credentials obtained through the SDK-supported login flow
- Optional: Node.js and `agent-browser` 0.33.x for browser-backed tools
- Optional: FFmpeg for the PCM audio mixer path

## Install

The OOPZ SDK is a Git submodule, so clone recursively:

```bash
git clone --recurse-submodules https://github.com/EnricLiu/cywl-oopz.git
cd cywl-oopz
uv sync --all-groups
```

For an existing clone without the submodule:

```bash
git submodule update --init --recursive
```

To enable browser-backed reading, install the supported browser client on the
bot host:

```bash
npm install -g agent-browser@0.33.0
agent-browser install
```

The bot starts and owns the MCP subprocess itself; no separately managed MCP
server is needed.

## Configure and start

1. Create local configuration:

   ```bash
   cp .env.example .env
   ```

2. Set the required values in `.env`:

   | Variable | Purpose |
   | --- | --- |
   | `OOPZ_DEVICE_ID` | Device identifier from the OOPZ SDK login flow |
   | `OOPZ_PERSON_UID` | Bot account person UID |
   | `OOPZ_JWT_TOKEN` | OOPZ authentication token |
   | `DATABASE_URL` | PostgreSQL connection URL |

3. Apply schema migrations:

   ```bash
   uv run alembic upgrade head
   ```

   Migrations are explicit deployment steps. The application checks database
   connectivity at startup but does not apply migrations automatically.

4. Choose a text conversation route:

   - **Legacy chat:** set `CYWL_CHAT_ENABLED=true` and configure the
     `CYWL_LLM_*` values for an OpenAI-compatible provider.
   - **Agent chat:** set `CYWL_AGENT_MODE=agent`, then configure enabled LLM
     provider and model records in PostgreSQL. One enabled
     application-default model is required; it must support tool calling when
     Agent tools are enabled.

5. Start the bot:

   ```bash
   uv run cywl-oopz
   ```

   The repository-root compatibility command is also available:

   ```bash
   python main.py
   ```

`.env.example` is the complete, authoritative list of runtime options and
defaults. It is safe to commit only as a template; `.env` is ignored and must
remain local.

## Feature switches

Most optional capabilities are disabled by default. Enable only the ones whose
dependencies and runtime configuration are ready.

| Setting | Default | Effect and additional requirement |
| --- | --- | --- |
| `CYWL_CHAT_ENABLED` | `false` | Enables legacy direct-LLM chat; requires `CYWL_LLM_*`. |
| `CYWL_AGENT_MODE` | `legacy` | Set to `agent` for the database-backed provider/model catalog. |
| `CYWL_MUSIC_ENABLED` | `false` | Requires a NeteaseCloudMusicApi-compatible endpoint. |
| `CYWL_AUDIO_MIXER_ENABLED` | `false` | Enables the PCM mixer path; requires a compatible FFmpeg binary. |
| `CYWL_WEB_SEARCH_ENABLED` | `true` | Enables DuckDuckGo search tools; no API key is required. |
| `CYWL_WEB_BROWSER_ENABLED` | `false` | Requires `agent-browser` on `PATH` and its browser installation. |
| `CYWL_WEB_BROWSER_INTERACTION_ENABLED` | `false` | Allows click/fill/press only when the browser feature and channel tool policy also permit them. |
| `CYWL_VOICE_ENABLED` | `false` | Enables experimental realtime voice commands; voice provider/model and channel policy live in PostgreSQL. |

Agent tool access is the intersection of the application allow-list
(`CYWL_AGENT_ENABLED_TOOLS`), enabled features, and applicable channel policy.
This is why enabling a feature does not automatically make every tool
available in every channel.

## Using the bot

Text chat can begin with `/chat <message>`, a bot mention, or a direct message.
Normal channel messages are handled only where the stored channel policy
enables ambient chat. Command availability changes with the enabled route and
features; `/help` always displays the commands registered in the running bot.

| Command | Availability | Description |
| --- | --- | --- |
| `/ping` | Always | Check whether the bot is online. |
| `/help` | Always | List commands registered by this deployment. |
| `/status` | Always | Show safe component-health status. |
| `/chat <text>` | Always registered | Start or continue a text conversation. |
| `/new` | Always registered | Clear the caller's current conversation context. |
| `/cancel` | Always registered | Cancel the caller's active text response. |
| `/chat-status` | Always registered | Show conversation metadata without showing its content. |
| `/model [name]` | Legacy or Agent mode | View or switch the applicable conversation model. |
| `/provider [name] [model]` | Agent mode | View, list, or switch the Agent provider/model. |
| `/tools` | Agent mode | List tools authorized for the current conversation. |
| `/tool <name> [JSON]` | Agent mode | Inspect a tool schema or invoke an authorized tool directly. |
| `/memory …` | Agent mode | View, save, disable, or delete the caller's long-term memory. |
| `/skills` | Agent mode with skills enabled | List skills available to the caller. |
| `/voice start\|stop\|status\|models\|model\|voice` | Voice enabled | Control experimental realtime voice conversation. |

`/voice` requires both `CYWL_VOICE_ENABLED=true` and valid voice configuration
and channel policy in PostgreSQL.

## Project layout

```text
src/cywl_oopz/
  application.py        Composition root and application lifecycle
  commands/             Command parsing, routing, and basic commands
  core/                 Shared errors, health, observability, and task helpers
  features/
    agent/              Agent loop, models, memory, skills, delegation, tools
    chat/               Legacy chat, history, rate limits, and streaming
    music/              Catalog, queues, playback state, and playlists
    voice/              Realtime voice sessions and task coordination
    web/                Search and browser domain services
    audio/              PCM mixing, buffering, and source coordination
  integrations/         OOPZ, Qwen, DuckDuckGo, and agent-browser adapters
  storage/              SQLAlchemy persistence and Alembic migrations
tests/                  Unit, integration-contract, and opt-in live tests
sdk/                    OOPZ SDK submodule
```

## Develop and verify

Run formatting, linting, and tests:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

The CI workflow checks formatting and linting while excluding the submodule,
runs tests while ignoring it, and verifies migrations against PostgreSQL:

```bash
uv run ruff format --check . --exclude sdk
uv run ruff check . --exclude sdk
uv run pytest --ignore sdk
uv run alembic upgrade head
```

Live OOPZ and Qwen tests are opt-in, credential-dependent tests controlled by
the `CYWL_RUN_LIVE_*` variables documented in `.env.example`; they are skipped
by default.

## Security and privacy

- Never commit OOPZ tokens, API keys, database passwords, or real credentials.
- Keep runtime secrets in local ignored environment files or in the
  PostgreSQL-backed runtime catalog; do not put them in normal logs.
- Web results and page content are external data. Browser sessions are isolated
  by conversation, and interactive actions require explicit opt-in.

## License

[MIT](LICENSE)
