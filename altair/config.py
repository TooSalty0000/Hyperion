"""Altair-specific configuration."""

import os
from dataclasses import dataclass, field
from typing import Optional

from shared.config import BaseAgentConfig, AgentRegistryConfig


@dataclass
class AltairConfig(BaseAgentConfig):
    """Configuration for the Altair bot."""

    # Permission settings
    permission_timeout: float = 300.0  # 5 minutes default

    # CLI settings
    default_cli_tool: str = "claude"  # claude, gemini, codex

    # Discord bot settings
    command_prefix: str = "!"

    # Terminal settings (for session creation)
    cli_command: str = "bash"
    terminal_cols: int = 120
    terminal_rows: int = 40
    terminal_backend: str = "auto"

    @classmethod
    def from_env(cls) -> 'AltairConfig':
        """Load Altair configuration from environment."""
        base = BaseAgentConfig.from_env("altair")

        return cls(
            agent_name="altair",
            discord_token=base.discord_token,
            bot_user_id=base.bot_user_id,
            allowed_user_id=base.allowed_user_id,
            llm_provider=base.llm_provider,
            llm_api_key=base.llm_api_key,
            llm_model=base.llm_model,
            utility_llm_provider=base.utility_llm_provider,
            utility_llm_api_key=base.utility_llm_api_key,
            utility_llm_model=base.utility_llm_model,
            redis_url=base.redis_url,
            workspace_dir=base.workspace_dir,
            agent_registry=base.agent_registry,
            main_channel_id=base.main_channel_id,
            permission_timeout=float(os.getenv("ALTAIR_PERMISSION_TIMEOUT", "300")),
            default_cli_tool=os.getenv("ALTAIR_DEFAULT_CLI_TOOL", "claude"),
            command_prefix=os.getenv("ALTAIR_COMMAND_PREFIX", "!"),
            cli_command=os.getenv("ALTAIR_CLI_COMMAND", os.getenv("CLI_COMMAND", "bash")),
            terminal_cols=int(os.getenv("ALTAIR_TERMINAL_COLS", os.getenv("TERMINAL_COLS", "120"))),
            terminal_rows=int(os.getenv("ALTAIR_TERMINAL_ROWS", os.getenv("TERMINAL_ROWS", "40"))),
            terminal_backend=os.getenv("ALTAIR_TERMINAL_BACKEND", os.getenv("TERMINAL_BACKEND", "auto")),
        )


# Singleton config instance
_config: Optional[AltairConfig] = None


def get_config() -> AltairConfig:
    """Get the Altair configuration singleton."""
    global _config
    if _config is None:
        _config = AltairConfig.from_env()
    return _config
