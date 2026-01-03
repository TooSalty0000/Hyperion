"""Canopus-specific configuration."""

import os
from dataclasses import dataclass
from typing import Optional

from shared.config import BaseAgentConfig


@dataclass
class CanopusConfig(BaseAgentConfig):
    """Configuration for the Canopus bot."""

    # Discord bot settings
    command_prefix: str = "!"

    # Browser settings
    browser_type: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720

    # Screenshot settings
    screenshot_dir: str = "./screenshots"
    screenshot_format: str = "png"  # png, jpeg

    # Timeouts (in milliseconds)
    navigation_timeout: int = 30000
    action_timeout: int = 10000

    # Session management
    max_sessions_per_channel: int = 1
    session_idle_timeout: int = 1800  # 30 minutes

    @classmethod
    def from_env(cls) -> 'CanopusConfig':
        """Load Canopus configuration from environment."""
        base = BaseAgentConfig.from_env("canopus")

        return cls(
            agent_name="canopus",
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
            command_prefix=os.getenv("CANOPUS_COMMAND_PREFIX", "!"),
            browser_type=os.getenv("CANOPUS_BROWSER", "chromium"),
            headless=os.getenv("CANOPUS_HEADLESS", "true").lower() == "true",
            viewport_width=int(os.getenv("CANOPUS_VIEWPORT_WIDTH", "1280")),
            viewport_height=int(os.getenv("CANOPUS_VIEWPORT_HEIGHT", "720")),
            screenshot_dir=os.getenv("CANOPUS_SCREENSHOT_DIR", "./screenshots"),
            screenshot_format=os.getenv("CANOPUS_SCREENSHOT_FORMAT", "png"),
            navigation_timeout=int(os.getenv("CANOPUS_NAVIGATION_TIMEOUT", "30000")),
            action_timeout=int(os.getenv("CANOPUS_ACTION_TIMEOUT", "10000")),
            max_sessions_per_channel=int(os.getenv("CANOPUS_MAX_SESSIONS", "1")),
            session_idle_timeout=int(os.getenv("CANOPUS_SESSION_IDLE_TIMEOUT", "1800")),
        )


# Singleton config instance
_config: Optional[CanopusConfig] = None


def get_config() -> CanopusConfig:
    """Get the Canopus configuration singleton."""
    global _config
    if _config is None:
        _config = CanopusConfig.from_env()
    return _config
