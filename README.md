# CYWL OOPZ Bot

基于 `oopz-sdk` 的异步 OOPZ 社区娱乐机器人。当前已支持 AI Agent 文字对话、
OOPZ 单消息流式显示、点歌播放，以及可选的公开网页搜索与浏览器工具；语音对话仍按路线图后续交付。
旧 KOOK 项目保留在 `ref/kook-dj`，仅作为功能参考。

## 当前功能

- OOPZ WebSocket 生命周期、自动订阅已加入的域、`/ping`、`/help` 与 `/status`。
- 基于 Pydantic AI 的 Agent loop，Provider 与模型由 PostgreSQL 管理并支持用户切换。
- 按对话持久化 thread、消息、摘要和记忆，带 timeout、取消、工具调用及并发预算。
- 每套 Agent loop 只创建一条 OOPZ 回复，通过消息编辑显示思考、工具步骤和正文，终态严格限制在
  OOPZ 2,000 UTF-16 单位以内。
- 网易云音乐目录搜索、按语音频道隔离的队列与自动续播。
- DuckDuckGo 公开搜索，以及由 agent-browser MCP 驱动、按 conversation 隔离的网页读取和
  click/fill/press；浏览器能力默认关闭并按频道白名单启用。
- PostgreSQL（SQLAlchemy 异步引擎、`asyncpg`、Alembic）持久化运行时数据和频道设置。

## 配置与启动

1. 安装 Python 3.13+ 与 [uv](https://docs.astral.sh/uv/)。
2. 创建本地配置：`cp .env.example .env`。填入 OOPZ 凭据与 PostgreSQL `DATABASE_URL`；此文件已被忽略，绝不提交。
3. Agent 模式下把 Provider 和 Model 写入 PostgreSQL，并设置 `CYWL_AGENT_MODE=agent`；
   legacy 模式仍可使用 `.env` 中的 `CYWL_LLM_*` 配置。
4. 显式执行数据库迁移：`uv run alembic upgrade head`。
5. 同步并运行：`uv sync --all-groups && uv run cywl-oopz`。

应用启动会以只读 `SELECT 1` 检查数据库；它不会自动执行迁移。

如需网页读取，另安装固定兼容版本及浏览器：

```bash
npm install -g agent-browser@0.33.0
agent-browser install
```

然后设置 `CYWL_WEB_BROWSER_ENABLED=true`。应用会自行启动和关闭 MCP 子进程；
`CYWL_WEB_BROWSER_INTERACTION_ENABLED=true` 只是第一层开关，click/fill/press 还需加入目标频道的
`enabled_agent_tools`。默认只启用 DuckDuckGo 搜索，浏览器读取和交互维持 opt-in。

## 命令

| 命令 | 说明 |
| --- | --- |
| `/ping` | 检查机器人是否在线。 |
| `/help` | 显示可用命令。 |
| `/status` | 显示 OOPZ、数据库和 LLM 的安全状态摘要。 |
| `/chat <内容>` | 发起或继续自己的文字对话。 |
| 提及机器人 + 内容 | 发起或继续自己的文字对话。 |
| 私聊普通消息 | 直接进行文字对话；仍使用独立的私聊会话。 |
| `/new` | 清空当前范围内、当前用户的会话历史。 |
| `/cancel` | 取消当前正在生成的回复，并等待相关资源释放。 |
| `/model [名称]` | 查看当前模型；切换需要发送者在 `CYWL_CHAT_MODEL_SELECTION_USERS` 中，且模型位于允许列表。 |
| `/chat-status` | 查看会话状态、模型、保留消息数与冷却时间，不显示聊天内容。 |

普通频道消息默认不会触发 LLM。阶段 1 已支持通过 `channel_settings.chat_enabled` 显式开启频道，但管理命令按路线图留在阶段 3；当前可通过受控数据库运维流程配置。

## 项目结构

```text
src/cywl_oopz/
  application.py       # 组合根与资源生命周期
  core/                # 配置错误、健康状态等通用能力
  commands/            # 命令协议和基础命令
  features/agent/      # Agent loop、Provider/模型目录、tools、memory
  features/chat/       # 对话命令、legacy chat 与通用 progress
  features/music/      # 搜索、队列和播放状态
  features/web/        # 搜索与浏览器领域服务
  integrations/web/    # DDGS 与 agent-browser MCP adapters
  storage/             # PostgreSQL 模型、Repository、Alembic 迁移
tests/                 # 不依赖真实凭据的单元与合约测试
impl-logs/             # 实施过程、设计抉择和验证记录
```

## 开发检查

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
DATABASE_URL=postgresql://... uv run alembic upgrade head
```

详细的分阶段规划、验收条件和风险清单见 [开发路线图](docs/development-roadmap.md)；LLM Chat
向 AI Agent 升级的架构、Agent loop、工具、memory 和迁移方案见
[AI Agent 设计](docs/ai-agent-design.md)。当前实现说明见
[阶段 0–1 实施日志](impl-logs/2026-07-26-stage-0-and-1-design.md)，Agent 规划过程与决定见
[Agent 规划日志](impl-logs/2026-07-26-ai-agent-planning.md)。联网搜索与浏览器工具的架构、分阶段实施和
验收见 [Web Tools 实施规划](docs/web-search-browser-tools-implementation-plan.md)。
