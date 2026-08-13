# CYWL OOPZ Bot

[English](README.md)

CYWL 是一个面向 [OOPZ](https://oopz.com) 社区的异步娱乐机器人。它以
CYWL——虚拟歌手初音未来（Hatsune Miku）的形象与用户互动，在同一进程中提供
对话、Agent 工具、音乐、联网研究和实验性的实时语音对话。

项目基于 OOPZ SDK、Python 异步 I/O 和 PostgreSQL 构建，并为各项功能保留可替换的
集成边界。`BotApplication` 是组合根，负责持有 OOPZ 客户端和所有服务的生命周期。

## 功能

- **稳定接入 OOPZ**：管理 WebSocket 客户端、订阅、重连和健康状态。
- **两种文字对话模式**：可使用直接连接 OpenAI 兼容接口的 LLM，也可使用
  PostgreSQL 驱动的 Pydantic AI Agent。两者都会保存作用域化的会话历史，并提供取消、
  超时和有界并发控制。
- **有状态的 Agent**：可按会话选择 Provider 和模型、压缩长对话、保存用户主动启用的
  记忆、加载持久化技能，并且只调用经策略授权的工具。可选地将 Agent 进度持续更新在
  一条 OOPZ 回复中。
- **语音频道音乐播放**：搜索网易云、YouTube 或 Bilibili，通过隔离的 `yt-dlp`
  worker 解析受支持的单曲链接，为每个语音频道维护串行队列，并管理混合来源歌单。
- **联网研究**：无需 API Key 即可使用 DuckDuckGo 搜索；可选用隔离的
  `agent-browser` MCP 会话阅读网页。浏览器交互还需要单独启用并受频道策略控制。
- **语音频道对话（实验性）**：借助 Qwen Omni/Audio 提供实时 STT/LLM/TTS 对话，拥有
  按频道划分的会话所有权、打断处理、空闲限制和委派 Agent 任务。
- **持久化运行状态**：通过 SQLAlchemy 和 Alembic 保存会话、Agent 运行记录、
  Provider/模型目录、记忆、技能、频道策略、歌单和语音会话。

## 环境要求

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL 14+ 或兼容的服务端
- 通过 SDK 支持的登录流程获得的 OOPZ 账号凭据
- 可选：用于网页工具的 Node.js 与 `agent-browser` 0.33.x
- 可选：用于 YouTube/Bilibili 音频的 FFmpeg 与 Deno 2.3+（或 Node.js 22+）

## 安装

OOPZ SDK 以 Git 子模块形式存在，因此请递归克隆：

```bash
git clone --recurse-submodules https://github.com/EnricLiu/cywl-oopz.git
cd cywl-oopz
uv sync --all-groups
```

如果已有克隆缺少子模块：

```bash
git submodule update --init --recursive
```

若要启用浏览器阅读功能，请在机器人主机上安装受支持的浏览器客户端：

```bash
npm install -g agent-browser@0.33.0
agent-browser install
```

机器人会自行启动并管理 MCP 子进程，无需另行部署 MCP 服务。

## 配置与启动

1. 创建本地配置：

   ```bash
   cp .env.example .env
   ```

2. 在 `.env` 中填写必需的值：

   | 变量 | 用途 |
   | --- | --- |
   | `OOPZ_DEVICE_ID` | 从 OOPZ SDK 登录流程获得的设备标识 |
   | `OOPZ_PERSON_UID` | 机器人账号的 person UID |
   | `OOPZ_JWT_TOKEN` | OOPZ 认证令牌 |
   | `DATABASE_URL` | PostgreSQL 连接 URL |

3. 执行数据库迁移：

   ```bash
   uv run alembic upgrade head
   ```

   迁移是显式的部署操作。应用启动时只检查数据库连通性，不会自动执行迁移。

4. 选择文字对话路径：

   - **Legacy chat**：设置 `CYWL_CHAT_ENABLED=true`，并配置指向 OpenAI 兼容
     Provider 的 `CYWL_LLM_*` 变量。
   - **Agent chat**：设置 `CYWL_AGENT_MODE=agent`，然后在 PostgreSQL 中配置已启用的
     LLM Provider 和模型记录。必须存在一个已启用的应用默认模型；当启用 Agent 工具时，
     它还必须支持工具调用。

5. 启动机器人：

   ```bash
   uv run cywl-oopz
   ```

   也可以在仓库根目录使用兼容入口：

   ```bash
   python main.py
   ```

### 进程托管与 `/reboot`

`/reboot` 会执行完整的应用级优雅停机，并以状态码 `75` 退出；它不会在进程内启动或替换自身。
若希望命令执行后 Bot 自动上线，需要用外部 supervisor 托管 CYWL。例如 systemd 可配置
`Restart=on-failure` 与 `RestartSec=2`，Docker/Compose 可使用 `restart: unless-stopped`。
OOPZ 正常断开时退出码为 `0`。

`.env.example` 是完整且权威的运行时选项与默认值清单。它可安全地作为模板提交；
`.env` 已被忽略，必须仅保存在本地。

## 功能开关

多数可选功能默认关闭。仅在依赖和运行时配置都准备好后再启用它们。

| 设置 | 默认值 | 效果与额外要求 |
| --- | --- | --- |
| `CYWL_CHAT_ENABLED` | `false` | 启用 Legacy 直接 LLM 对话；需要 `CYWL_LLM_*`。 |
| `CYWL_AGENT_MODE` | `legacy` | 设置为 `agent` 以使用数据库中的 Provider/模型目录。 |
| `CYWL_MUSIC_ENABLED` | `false` | 启用 `CYWL_MUSIC_SOURCES` 中配置的来源；网易云需要 API 服务，YouTube/Bilibili 需要 PCM 混音器、FFmpeg 和受支持的 JavaScript runtime。 |
| `CYWL_AUDIO_MIXER_ENABLED` | `false` | 启用 PCM 混音路径；需要兼容的 FFmpeg 二进制文件。 |
| `CYWL_WEB_SEARCH_ENABLED` | `true` | 启用 DuckDuckGo 搜索工具；不需要 API Key。 |
| `CYWL_WEB_BROWSER_ENABLED` | `false` | 需要 `PATH` 中存在 `agent-browser`，并已安装其浏览器。 |
| `CYWL_WEB_BROWSER_INTERACTION_ENABLED` | `false` | 仅当浏览器功能和频道工具策略均允许时，才可点击、填写和按键。 |
| `CYWL_VOICE_ENABLED` | `false` | 启用实验性实时语音命令；语音 Provider/模型和频道策略保存在 PostgreSQL。 |

Agent 工具权限取决于应用级白名单（`CYWL_AGENT_ENABLED_TOOLS`）、已启用功能和适用频道
策略三者的交集。因此，启用某项功能并不意味着所有频道都会自动拥有该功能的全部工具。

## 使用机器人

可通过 `/chat <消息>`、提及机器人或私信发起文字对话。普通频道消息只有在保存的频道策略
启用了 ambient chat 时才会被处理。命令是否可用取决于已启用的路径和功能；`/help` 始终会
显示当前运行实例实际注册的命令。

| 命令 | 可用条件 | 说明 |
| --- | --- | --- |
| `/ping` | 始终 | 检查机器人是否在线。 |
| `/help` | 始终 | 列出当前实例注册的命令。 |
| `/status` | 始终 | 显示不会泄露配置的组件健康状态。 |
| `/chat <文本>` | 始终注册 | 发起或继续文字对话。 |
| `/new` | 始终注册 | 清空调用者当前对话的上下文。 |
| `/cancel` | 始终注册 | 取消调用者当前正在生成的文字回复。 |
| `/chat-status` | 始终注册 | 查看不含对话正文的会话元数据。 |
| `/model [名称]` | Legacy 或 Agent 模式 | 查看或切换当前会话适用的模型。 |
| `/provider [名称] [模型]` | Agent 模式 | 查看、列出或切换 Agent Provider/模型。 |
| `/tools` | Agent 模式 | 列出当前会话获授权的工具。 |
| `/tool <名称> [JSON]` | Agent 模式 | 查看工具 Schema，或直接调用获授权的工具。 |
| `/memory …` | Agent 模式 | 查看、保存、停用或删除调用者的长期记忆。 |
| `/skills` | Agent 模式且已启用技能 | 列出调用者可用的技能。 |
| `/whoami`、`/role …` | 取决于 RBAC | 查看调用者身份，并管理有作用域的角色。 |
| `/init [channel\|area]` | 获授权管理员 | 初始化缺失的频道配置，不覆盖已有值。 |
| `/debug [-v\|--verbose]` | 获授权管理员 | 将引用的 Agent 回复展开为有界诊断分页。 |
| `/recall` | 获授权版主/管理员 | 撤回引用的一条 CYWL 自有消息。 |
| `/reboot` | 全局 owner/admin | 优雅退出并返回状态码 75，由外部 supervisor 重启。 |
| `/music …` | 已启用音乐 | 按关键词或 URL 搜索/点歌，查看来源和队列，设置模式并管理 area 歌单。文字查询可在内容前用 `--source youtube\|bilibili\|netease` 覆盖默认来源。 |
| `/voice start\|stop\|status\|models\|model\|voice` | 已启用语音 | 控制实验性实时语音对话。 |

`/voice` 同时需要 `CYWL_VOICE_ENABLED=true`，以及 PostgreSQL 中有效的语音配置和频道策略。

多来源音乐通过 `CYWL_MUSIC_SOURCES` 和 `CYWL_MUSIC_DEFAULT_SOURCE` 配置。普通文字查询
使用默认来源；受支持的网易云、YouTube、Bilibili 与 `b23.tv` 单曲 URL 会自动识别。
YouTube/Bilibili 提取在有界子进程中运行，并要求 `CYWL_AUDIO_MIXER_ENABLED=true`。
需要登录态的内容可选用浏览器 Cookie 导出文件，但不要将这些文件提交到 Git。

## 项目结构

```text
src/cywl_oopz/
  application.py        组合根与应用生命周期
  commands/             命令解析、路由和基础命令
  core/                 共享错误、健康检查、可观测性和任务辅助类
  features/
    agent/              Agent 循环、模型、记忆、技能、委派和工具
    chat/               Legacy 对话、历史记录、限流和流式输出
    music/              曲库、队列、播放状态和歌单
    voice/              实时语音会话与任务协调
    web/                搜索和浏览器领域服务
    audio/              PCM 混音、缓冲与音源协调
  integrations/         OOPZ、Qwen、DuckDuckGo 和 agent-browser 适配器
  storage/              SQLAlchemy 持久化层和 Alembic 迁移
tests/                  单元测试、集成契约测试和可选实时测试
sdk/                    OOPZ SDK 子模块
```

## 开发与验证

执行格式化、静态检查和测试：

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

CI 会在排除子模块的情况下检查格式和静态规则、运行测试，并在 PostgreSQL 上验证迁移：

```bash
uv run ruff format --check . --exclude sdk
uv run ruff check . --exclude sdk
uv run pytest --ignore sdk
uv run alembic upgrade head
```

YouTube 与 Bilibili PCM 验收默认跳过，且不会加入 OOPZ 语音频道；需要联网验收时运行：

```bash
CYWL_RUN_LIVE_YOUTUBE_PCM_TESTS=1 uv run pytest tests/test_youtube_music_live.py
CYWL_RUN_LIVE_BILIBILI_PCM_TESTS=1 uv run pytest tests/test_bilibili_music_live.py
```

升级 `yt-dlp` 时应更新 `uv.lock`，先运行 provider/worker 单元测试，再用部署环境配置的
JavaScript runtime 与 FFmpeg 跑完这两项公网 PCM gate，最后再发布新的 lock。

真实 OOPZ 与 Qwen 测试是可选的、依赖凭据的测试，由 `.env.example` 中说明的
`CYWL_RUN_LIVE_*` 变量控制；默认会跳过。

## 安全与隐私

- 切勿提交 OOPZ 令牌、API Key、数据库密码或任何真实凭据。
- 将运行时密钥保存在本地忽略的环境文件或 PostgreSQL 运行时目录中，不要写入常规日志。
- 搜索结果和网页内容都属于外部数据。浏览器会话按对话隔离，交互操作必须显式启用。

## 许可证

[MIT](LICENSE)
