# Hyperion Project Index

> **Project Name**: Hyperion (Vega & Altair Multi-Agent System)
> **Generated**: 2025-12-30
> **Type**: Discord Bot Multi-Agent System

---

## Overview

Hyperion is a **multi-agent Discord bot system** consisting of two separate Discord bots that communicate via visible @mentions:

- **Vega** - The head conversational agent (handles conversation, planning, delegation)
- **Altair** - The autonomous project manager (handles terminal sessions, CLI tools, execution)

The system bridges Discord messages to local terminal processes (shell or CLI agents like Claude Code, Gemini CLI), supporting multiple concurrent sessions using tmux with dedicated Discord channels per session.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         Discord Server                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    @mentions    ┌──────────────┐             │
│   │    Vega      │◄───────────────►│   Altair     │             │
│   │ (Converser)  │                 │ (Executor)   │             │
│   └──────┬───────┘                 └──────┬───────┘             │
│          │                                │                      │
│   User messages                    Terminal sessions             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │         shared/               │
                    │  ├── llm/         (Providers) │
                    │  ├── session/     (Registry)  │
                    │  ├── terminal/    (Backends)  │
                    │  ├── tools/       (Framework) │
                    │  ├── database/    (Storage)   │
                    │  └── channel/     (Discord)   │
                    └───────────────────────────────┘
```

---

## Directory Structure

```
Hyperion/
├── bot.py                    # Legacy entry point (Vega only)
├── run_all.py                # Main entry point (both bots)
├── Makefile                  # Build/run commands
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── CLAUDE.md                 # Project instructions
├── README.md                 # User-facing documentation
│
├── vega/                     # Vega bot package
│   ├── __init__.py
│   ├── config.py             # Vega configuration
│   ├── agents/
│   │   └── vega.py           # VegaAgent implementation
│   ├── cogs/
│   │   └── core.py           # VegaCore Discord cog
│   └── tools/
│       ├── mention.py        # Inter-agent @mention tool
│       └── delegation.py     # Project summary tool
│
├── altair/                   # Altair bot package
│   ├── __init__.py
│   ├── bot.py                # AltairBot class
│   ├── config.py             # Altair configuration
│   ├── agent.py              # AltairAgent implementation
│   ├── permission.py         # Permission request system
│   ├── cogs/
│   │   └── core.py           # AltairCore Discord cog
│   └── tools/
│       ├── cli_commands.py   # CLI interaction tools
│       ├── session.py        # Session management tools
│       └── monitoring.py     # Active monitoring tools
│
└── shared/                   # Shared infrastructure
    ├── __init__.py
    ├── base_agent.py         # BaseAgent ABC
    ├── base_bot.py           # Bot utilities
    ├── config.py             # Shared config helpers
    ├── utils.py              # VirtualScreen terminal emulator
    ├── agent_messaging.py    # Inter-agent messaging
    │
    ├── llm/                  # LLM Provider abstraction
    │   ├── interface.py      # LLMProvider ABC
    │   ├── factory.py        # Provider factory
    │   ├── types.py          # Message, Tool types
    │   ├── gemini.py         # Gemini implementation
    │   ├── openai.py         # OpenAI implementation
    │   └── anthropic.py      # Anthropic implementation
    │
    ├── session/              # Session management
    │   ├── models.py         # ActiveSession dataclass
    │   └── registry.py       # SessionRegistry
    │
    ├── terminal/             # Terminal backends
    │   ├── interface.py      # TerminalBackend ABC
    │   ├── factory.py        # Backend factory
    │   ├── tmux.py           # Tmux backend (preferred)
    │   └── pexpect_backend.py # Pexpect fallback
    │
    ├── tools/                # Tool framework
    │   ├── interface.py      # Tool ABC, ToolContext
    │   └── registry.py       # ToolRegistry
    │
    ├── database/             # Session persistence
    │   ├── interface.py      # SessionStore ABC
    │   ├── memory.py         # In-memory implementation
    │   └── redis.py          # Redis implementation
    │
    ├── channel/              # Discord channel management
    │   └── manager.py        # ChannelManager
    │
    ├── project/              # Project management
    │   ├── manager.py        # ProjectManager
    │   ├── models.py         # Project model
    │   └── status.py         # Status tracking
    │
    └── conversation/         # Conversation history
        ├── manager.py        # ConversationManager
        └── models.py         # Conversation models
```

---

## Core Components

### 1. Agent System

#### BaseAgent (`shared/base_agent.py`)
Abstract base class for all agents. Defines:
- `AgentContext` - Execution context (channel_id, user_id, message_content, etc.)
- `AgentResponse` - Unified response format
- Abstract methods: `process()`, `should_handle()`, `system_prompt`

#### VegaAgent (`vega/agents/vega.py`)
The conversational head agent:
- **Persona**: Witty, strategic, confident but humble
- **Role**: Primary conversational interface, delegates hands-on work to Altair
- **Tools**: `mention_agent`, `get_project_summary`
- **Patterns**: Questions, explanations, status inquiries

#### AltairAgent (`altair/agent.py`)
The autonomous project executor:
- **Persona**: Methodical, precise, autonomous execution
- **Role**: Manages terminal sessions, CLI tools (Claude Code, etc.)
- **Tools**: Session management, CLI commands, monitoring
- **Patterns**: Run/execute/build/deploy commands

### 2. LLM Provider System

#### LLMProvider Interface (`shared/llm/interface.py`)
Abstract interface supporting:
- `complete()` - Generate completion with optional tool calls
- `stream()` - Streaming generation
- Message/tool format conversion

#### Supported Providers
| Provider | Implementation | Default Model |
|----------|---------------|---------------|
| Gemini | `gemini.py` | `gemini-2.0-flash-exp` |
| OpenAI | `openai.py` | `gpt-4o` |
| Anthropic | `anthropic.py` | `claude-sonnet-4-20250514` |

### 3. Session Management

#### SessionRegistry (`shared/session/registry.py`)
Central manager for all CLI sessions:
- Creates sessions with TerminalBackend + VirtualScreen
- Health check detects dead processes
- Channel-to-session mapping

#### Terminal Backends
| Backend | File | Features |
|---------|------|----------|
| TmuxBackend | `tmux.py` | Local attachment, robust |
| PexpectBackend | `pexpect_backend.py` | Fallback, no local attach |

### 4. Tool Framework

#### Tool Interface (`shared/tools/interface.py`)
- `Tool` ABC - Define name, description, parameters, execute()
- `ToolContext` - Provides session_registry, project_manager, permission_manager, etc.
- `ToolRegistry` - Register and execute tools

#### Vega Tools
| Tool | Description |
|------|-------------|
| `mention_agent` | Send @mention to another agent |
| `get_project_summary` | Get project overview |

#### Altair Tools
| Category | Tools |
|----------|-------|
| Session | `create_session`, `terminate_session`, `switch_session`, `list_projects` |
| CLI | `start_claude_code`, `send_cli_command`, `get_cli_output`, `wait_for_completion`, `get_session_status`, `list_active_sessions` |
| Monitoring | `monitor_session`, `check_for_prompt`, `respond_to_prompt` |

### 5. Discord Integration

#### VegaCore Cog (`vega/cogs/core.py`)
- Message routing (plain text, !vli commands, !vp commands)
- Session lifecycle management
- Output loop for terminal display

#### AltairCore Cog (`altair/cogs/core.py`)
- @mention handling
- Permission system integration
- Session output loops

#### ChannelManager (`shared/channel/manager.py`)
- Creates Discord channels for sessions
- Organizes under "Vega CLI Sessions" category
- Channel cleanup on session termination

---

## Discord Commands

### Vega Commands

| Command | Description |
|---------|-------------|
| `!v help` | Show help |
| `!vli` | Start new CLI session |
| `!vli <project>` | Start session in project workspace |
| `!vli ls` | List active sessions |
| `!vli quit` | Send Ctrl+C |
| `!vli exit [id]` | Terminate session |
| `!vli status` | Show session status |
| `!vli attach` | Show tmux attach command |
| `!vli resize <cols> <rows>` | Resize terminal |
| `!vp ls` | List projects |
| `!vp <name> [path]` | Register project |
| `!vp rm <name>` | Remove project |

### Altair Commands

| Command | Description |
|---------|-------------|
| `!status` | Get Altair's current status |
| `!sessions` | List active terminal sessions |
| `@Altair <message>` | Invoke Altair for task |

---

## Configuration

### Environment Variables

```bash
# Discord
DISCORD_TOKEN=              # Vega bot token
ALTAIR_DISCORD_TOKEN=       # Altair bot token
ALLOWED_USER_ID=            # Authorized user ID

# Agent Registry (for @mentions)
AGENT_VEGA_USER_ID=         # Vega's Discord user ID
AGENT_ALTAIR_USER_ID=       # Altair's Discord user ID

# LLM Configuration
LLM_PROVIDER=gemini         # gemini|openai|anthropic
LLM_API_KEY=                # API key
LLM_MODEL=                  # Optional model override

# Terminal Configuration
CLI_COMMAND=zsh             # Default command
WORKSPACE_DIR=              # Working directory
TERMINAL_COLS=80
TERMINAL_ROWS=24
TERMINAL_BACKEND=tmux       # tmux|pexpect

# Optional
REDIS_URL=                  # Enable persistence
AGENT_MAX_ITERATIONS=20     # LLM loop limit
AGENT_TEMPERATURE=0.7       # LLM temperature
```

---

## Data Flow

### Message Processing (Vega)

```
User message in Discord
        │
        ▼
on_message() → Check author, commands
        │
        ├─► !vli commands → Session management
        ├─► !vp commands  → Project management
        ├─► CLI channel   → Raw passthrough to terminal
        │
        └─► Agent channel → VegaAgent.process()
                │
                ├─► Build messages (system prompt + history)
                ├─► LLM.complete() with tools
                ├─► Execute tool calls if any
                └─► Return AgentResponse
```

### Session Lifecycle

```
1. !vli command
        │
        ▼
2. ChannelManager.create_channel()
        │
        ▼
3. SessionRegistry.create_session()
        │
        ├─► TerminalConfig
        ├─► create_terminal_backend()
        ├─► terminal.start()
        └─► VirtualScreen()
        │
        ▼
4. Output loop → Capture pane → VirtualScreen → Discord message
        │
        ▼
5. User input → terminal.send_input()
        │
        ▼
6. !vli exit → terminate_session() → delete_channel()
```

### Inter-Agent Communication

```
Vega receives task requiring execution
        │
        ▼
Vega uses mention_agent tool
        │
        ▼
Discord message: "@Altair [request]"
        │
        ▼
Altair's on_message() → mentioned?
        │
        ▼
AltairAgent.process()
        │
        ├─► Create session if needed
        ├─► Send CLI commands
        ├─► Monitor progress
        └─► Report results
```

---

## Key Patterns

### Agent Loop Pattern
Both agents use a similar tool-calling loop:
```python
for _ in range(max_iterations):
    response = await llm.complete(messages, tools)
    if not response.tool_calls:
        return AgentResponse(content=response.content)
    results = await tools.execute_batch(response.tool_calls, context)
    messages.append(assistant_message)
    messages.append(tool_results_message)
```

### Permission System (Altair)
Before sensitive operations:
```python
approved = await permission_manager.request_permission(
    channel, user_id, action_description, timeout
)
if not approved:
    return "Permission denied or timed out"
```

### Terminal Backend Abstraction
```python
terminal = create_terminal_backend(config, preferred_backend="tmux")
await terminal.start()
await terminal.send_input(text)
output = await terminal.get_output()
await terminal.kill()
```

---

## Dependencies

```
discord.py          # Discord API
pexpect             # Terminal interaction
psutil              # Process monitoring
python-dotenv       # Environment loading
google-genai        # Gemini LLM provider
redis               # Optional persistence
```

---

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Configure environment
cp .env.example .env
# Edit .env with your tokens and settings

# 3. Run both bots
make start

# Or run individually
make start-vega
make start-altair
```

---

## Bot Permissions Required

- Manage Channels (create/delete session channels)
- Send Messages
- Read Messages
- Manage Messages (for UI mode)
- Add Reactions (permission system)

---

## Extension Points

### Adding a New LLM Provider
1. Create `shared/llm/<provider>.py` implementing `LLMProvider`
2. Add to factory in `shared/llm/factory.py`
3. Add default model to `DEFAULT_MODELS`

### Adding a New Agent Tool
1. Create tool class extending `Tool` in appropriate package
2. Implement `name`, `description`, `parameters`, `execute()`
3. Register in agent's `_register_tools()` method

### Adding a New Agent
1. Create agent class extending `BaseAgent`
2. Implement required abstract methods
3. Create corresponding Discord cog
4. Add to `run_all.py`

---

## File Quick Reference

| File | Purpose | Lines |
|------|---------|-------|
| `run_all.py` | Multi-bot launcher | ~200 |
| `vega/agents/vega.py` | Vega agent logic | ~320 |
| `vega/cogs/core.py` | Vega Discord integration | ~700 |
| `altair/agent.py` | Altair agent logic | ~540 |
| `altair/cogs/core.py` | Altair Discord integration | ~400 |
| `shared/base_agent.py` | Agent base class | ~120 |
| `shared/llm/interface.py` | LLM abstraction | ~125 |
| `shared/session/registry.py` | Session management | ~200 |
| `shared/tools/interface.py` | Tool framework | ~120 |

---

*This documentation was auto-generated by analyzing the Hyperion codebase.*
