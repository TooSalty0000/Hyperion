"""Vega-specific configuration."""

import os
from dataclasses import dataclass
from typing import Optional

from shared.config import BaseAgentConfig


@dataclass
class VegaConfig(BaseAgentConfig):
    """Configuration for the Vega bot."""

    # Terminal settings (for session creation)
    cli_command: str = "zsh"
    terminal_cols: int = 80
    terminal_rows: int = 24
    terminal_backend: str = "tmux"

    @classmethod
    def from_env(cls) -> 'VegaConfig':
        """Load Vega configuration from environment."""
        base = BaseAgentConfig.from_env("vega")

        return cls(
            agent_name="vega",
            discord_token=base.discord_token or os.getenv("DISCORD_TOKEN", ""),
            bot_user_id=base.bot_user_id,
            allowed_user_id=base.allowed_user_id or int(os.getenv("ALLOWED_USER_ID", "0")),
            llm_provider=base.llm_provider,
            llm_api_key=base.llm_api_key or os.getenv("GEMINI_API_KEY", ""),
            llm_model=base.llm_model,
            utility_llm_provider=base.utility_llm_provider,
            utility_llm_api_key=base.utility_llm_api_key,
            utility_llm_model=base.utility_llm_model,
            redis_url=base.redis_url,
            workspace_dir=base.workspace_dir or os.getenv("WORKSPACE_DIR", "/tmp/workspace"),
            agent_registry=base.agent_registry,
            main_channel_id=base.main_channel_id,
            cli_command=os.getenv("CLI_COMMAND", "zsh"),
            terminal_cols=int(os.getenv("TERMINAL_COLS", "80")),
            terminal_rows=int(os.getenv("TERMINAL_ROWS", "24")),
            terminal_backend=os.getenv("TERMINAL_BACKEND", "tmux"),
        )


# Singleton config instance
_config: Optional[VegaConfig] = None


def get_config() -> VegaConfig:
    """Get the Vega configuration singleton."""
    global _config
    if _config is None:
        _config = VegaConfig.from_env()
    return _config


# Legacy compatibility - Config class for backwards compatibility
class Config:
    """Legacy configuration class for backwards compatibility.

    Deprecated: Use get_config() instead.
    """
    _instance: Optional[VegaConfig] = None

    @classmethod
    def _get(cls) -> VegaConfig:
        if cls._instance is None:
            cls._instance = get_config()
        return cls._instance

    @property
    def DISCORD_TOKEN(self) -> str:
        return self._get().discord_token

    @property
    def ALLOWED_USER_ID(self) -> int:
        return self._get().allowed_user_id

    @property
    def CLI_COMMAND(self) -> str:
        return self._get().cli_command

    @property
    def WORKSPACE_DIR(self) -> str:
        return self._get().workspace_dir

    @property
    def TERMINAL_COLS(self) -> int:
        return self._get().terminal_cols

    @property
    def TERMINAL_ROWS(self) -> int:
        return self._get().terminal_rows

    @property
    def TERMINAL_BACKEND(self) -> str:
        return self._get().terminal_backend

    @property
    def REDIS_URL(self) -> Optional[str]:
        return self._get().redis_url

    @property
    def LLM_PROVIDER(self) -> str:
        return self._get().llm_provider

    @property
    def LLM_API_KEY(self) -> str:
        return self._get().llm_api_key

    @property
    def LLM_MODEL(self) -> Optional[str]:
        return self._get().llm_model


# Create a singleton instance for backwards compatibility
Config = type('Config', (), {
    '_cfg': None,
    '__getattribute__': lambda self, name: (
        getattr(get_config(), {
            'DISCORD_TOKEN': 'discord_token',
            'ALLOWED_USER_ID': 'allowed_user_id',
            'CLI_COMMAND': 'cli_command',
            'WORKSPACE_DIR': 'workspace_dir',
            'TERMINAL_COLS': 'terminal_cols',
            'TERMINAL_ROWS': 'terminal_rows',
            'TERMINAL_BACKEND': 'terminal_backend',
            'REDIS_URL': 'redis_url',
            'LLM_PROVIDER': 'llm_provider',
            'LLM_API_KEY': 'llm_api_key',
            'LLM_MODEL': 'llm_model',
            'AGENT_CHANNEL_ID': 'main_channel_id',
        }.get(name, name))
        if name not in ('__class__', '__dict__', '_cfg')
        else object.__getattribute__(self, name)
    )
})()
