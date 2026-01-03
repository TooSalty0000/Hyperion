# Hyperion - Multi-Agent Discord System

Hyperion is a multi-agent AI system where specialized bots collaborate via Discord to accomplish tasks. Each agent has its own expertise and they communicate through @mentions, just like human team members.

## Agents

| Agent | Role | Specialization |
|-------|------|----------------|
| **Vega** | Conversational Lead | General conversation, strategy, delegation |
| **Altair** | Project Manager | Terminal sessions, CLI tools, code execution |
| **Polaris** | Calendar Specialist | Scheduling, events, reminders, time management |
| **Canopus** | Web Researcher | Browser automation, web search, data extraction |

## Key Features

- **Independent Operation**: Each bot runs independently (can be on separate machines)
- **Discord Communication**: Agents communicate via @mentions in Discord
- **Chime-In System**: Agents proactively respond to relevant messages without being explicitly mentioned
- **Automatic Handoffs**: Agents can hand off findings to the right specialist automatically
- **Shared State**: Redis-backed project management, memory, and conversations

---

# Vega - Discord CLI Bridge (Legacy Docs)

Vega is a Discord bot that acts as a secure bridge to your local terminal, allowing you to interact with CLI agents (like Claude, Gemini, or standard shells) remotely from your phone.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Configuration**:
    Copy `.env.example` to `.env` and fill in your details:
    ```bash
    cp .env.example .env
    ```
    - `DISCORD_TOKEN`: Your bot token from the [Discord Developer Portal](https://discord.com/developers/applications).
        > **IMPORTANT**: In the "Bot" tab of the Developer Portal, scroll down to **Privileged Gateway Intents** and enable **MESSAGE CONTENT INTENT**. The bot cannot read your commands without this!
    - `ALLOWED_USER_ID`: Your personal Discord User ID.
        1. Open Discord Settings -> Advanced -> Enable **Developer Mode**.
        2. Right-click your own Avatar/Name in any chat or the member list.
        3. Click **Copy User ID** (or just **Copy ID**).
    - `CLI_COMMAND`: The command to run. Defaults to `zsh`. To run an agent, set it to something like `claude` or `python my_agent.py`.
    - `WORKSPACE_DIR`: (Optional) The directory to start the process in. Useful for scoping the agent to a specific project.

3.  **Run the Bot**:
    ```bash
    python bot.py
    ```

## Usage

*   **`!v [command]`** or **`!vega [command]`**: Send input to the running process.
    *   Example: `!v ls -la`
    *   Example: `!v Please write a poem about rust.`
*   **`!v status`**: Check if the process is running and view host stats.
*   **`!v reset`**: Kill and restart the CLI process.

## Handling Interactive CLIs

Vega uses `pexpect` to maintain a persistent session.
- **Yes/No Prompts**: If the CLI asks `Do you want to continue? [y/n]`, you simply type `!v y` in Discord.
- **Streaming**: Output is buffered and sent to Discord in chunks.
- **ANSI Colors**: Terminal colors are automatically stripped to keep Discord messages clean.
