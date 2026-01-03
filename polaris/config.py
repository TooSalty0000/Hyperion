"""Polaris-specific configuration."""

import os
from dataclasses import dataclass, field
from typing import Optional

from shared.config import BaseAgentConfig, AgentRegistryConfig


@dataclass
class PolarisConfig(BaseAgentConfig):
    """Configuration for the Polaris bot."""

    # Discord bot settings
    command_prefix: str = "!"

    # Google Calendar settings
    google_credentials_path: Optional[str] = None
    google_token_path: Optional[str] = None  # Will default to ~/.hyperion/polaris/token.json
    default_calendar_id: str = "primary"

    # Scheduling defaults
    default_event_duration_minutes: int = 60
    working_hours_start: int = 9  # 9 AM
    working_hours_end: int = 17  # 5 PM
    timezone: str = "UTC"

    @classmethod
    def from_env(cls) -> 'PolarisConfig':
        """Load Polaris configuration from environment."""
        base = BaseAgentConfig.from_env("polaris")

        return cls(
            agent_name="polaris",
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
            command_prefix=os.getenv("POLARIS_COMMAND_PREFIX", "!"),
            google_credentials_path=os.getenv("GOOGLE_CREDENTIALS_PATH"),
            google_token_path=os.getenv("GOOGLE_TOKEN_PATH"),  # Defaults to ~/.hyperion/polaris/token.json in utils
            default_calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "primary"),
            default_event_duration_minutes=int(
                os.getenv("POLARIS_DEFAULT_EVENT_DURATION", "60")
            ),
            working_hours_start=int(os.getenv("POLARIS_WORKING_HOURS_START", "9")),
            working_hours_end=int(os.getenv("POLARIS_WORKING_HOURS_END", "17")),
            timezone=os.getenv("POLARIS_TIMEZONE", "UTC"),
        )


# Singleton config instance
_config: Optional[PolarisConfig] = None


def get_config() -> PolarisConfig:
    """Get the Polaris configuration singleton."""
    global _config
    if _config is None:
        _config = PolarisConfig.from_env()
    return _config
