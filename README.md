# Hyperion - Multi-Agent Discord System

Hyperion is a multi-agent AI system where specialized bots collaborate via Discord to accomplish tasks. Each agent has its own expertise and they communicate through @mentions, just like human team members.

## Agents

| Agent | Role | Specialization |
|-------|------|----------------|
| **Vega** | Conversational Lead | General conversation, strategy, delegation |
| **Altair** | Project Manager | Terminal sessions, CLI tools, code execution, ngrok tunnels |
| **Polaris** | Calendar Specialist | Google Calendar, scheduling, events, time management |
| **Canopus** | Web Researcher | Playwright browser automation, web search, data extraction |

## Key Features

- **Independent Operation**: Each bot runs independently (can be on separate machines)
- **Discord Communication**: Agents communicate via @mentions in Discord
- **Chime-In System**: Agents proactively respond to relevant messages without being explicitly mentioned
- **Automatic Handoffs**: Agents can hand off findings to the right specialist automatically
- **Shared State**: Redis-backed project management, memory, and conversations
- **Permission System**: Altair requests human approval for sensitive operations
- **Local Terminal Attachment**: Tmux sessions can be attached to locally while controlled from Discord

## Quick Start

### 1. Install Dependencies

```bash
make install
# or: pip install -r requirements.txt

# For Canopus browser automation:
playwright install chromium
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Each agent needs its own Discord bot token. See `.env.example` for the full configuration reference including:
- Discord bot tokens (one per agent)
- LLM provider settings (Gemini, OpenAI, or Anthropic)
- Agent-specific settings (terminal, calendar, browser)
- Redis connection (optional, enables persistence)

### 3. Run the Bots

```bash
# Start all bots
make start
# or: python run_all.py

# Start specific bots
python run_all.py --vega --altair
python run_all.py --polaris
make start-vega
make start-altair
```

## Discord Commands

### Altair Terminal Commands (`!vli`)

- `!vli` - Start new CLI session (creates a dedicated channel)
- `!vli ls` - List all active sessions
- `!vli quit` - Send Ctrl+C to current session
- `!vli exit [id]` - Terminate session (deletes channel)
- `!vli ctrl-d` / `!vli ctrl-z` / `!vli ctrl-l` - Control keys
- `!vli status` - Show current session status
- `!vli attach` - Show command to attach locally (tmux)
- `!vli resize <cols> <rows>` - Resize terminal
- `!vli mode <scroll|ui>` - Change display mode
- `!vli <key>` - Send special key (up, down, enter, esc)
- **Plain text in session channel** - Sends directly to process (no prefix needed)

### Local Terminal Attachment

When using the tmux backend, you can attach to sessions locally:
```bash
tmux attach -t vega-1  # Attach to session #1
```

This allows both Discord control AND local terminal control simultaneously.

## Deployment

### Local (Recommended for Development)

```bash
python run_all.py  # All bots in one process with graceful shutdown
```

### Docker Compose (Full System)

```bash
docker-compose up -d              # Start Redis + all 4 bots
docker-compose up -d vega altair  # Start specific services
docker-compose logs -f vega       # Follow logs
docker-compose down               # Stop everything
```

### Distributed (Across Machines)

Each bot can run on a separate machine using `docker-compose.single-bot.yml`. Agents communicate only via Discord messages, so no shared network is required beyond Discord API access.

## Architecture Overview

```
run_all.py              # Entry point - starts all bots concurrently
vega/                   # Vega agent (conversational lead)
altair/                 # Altair agent (CLI/project specialist)
polaris/                # Polaris agent (calendar specialist)
canopus/                # Canopus agent (web/browser specialist)
shared/                 # Common framework shared by all agents
  base_agent.py         #   BaseAgent ABC - core agent loop
  llm/                  #   LLM provider abstraction (Gemini, OpenAI, Anthropic)
  tools/                #   Tool framework (registry, interface, context)
  terminal/             #   Terminal backends (tmux, pexpect)
  session/              #   Session lifecycle management
  channel/              #   Discord channel management
  memory/               #   Redis-backed persistent memory
  project/              #   Project tracking and status
  conversation/         #   Conversation history management
  collaboration/        #   Chime-in system and suppression
  database/             #   Session stores (memory, file, Redis)
  events/               #   Event dispatcher for agent events
docker-compose.yml      # Full deployment (Redis + 4 bots)
```

## Bot Permissions Required

- Manage Channels (create/delete session channels)
- Send Messages
- Read Messages / Message Content Intent
- Manage Messages (edit for UI mode)
- Add Reactions (acknowledgment protocol)

## Dependencies

- Python 3.10+
- discord.py
- google-genai (Gemini LLM - primary provider)
- pexpect, psutil (terminal management)
- playwright (Canopus browser automation)
- redis (optional - persistence)
- httpx (ngrok tunnel management)
