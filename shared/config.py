"""Shared configuration for multi-agent system."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AgentRegistryConfig:
    """
    Configuration for the agent registry.

    Maps agent names to their Discord bot user IDs.
    This allows agents to @mention each other.
    """
    agents: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> 'AgentRegistryConfig':
        """Load agent registry from environment variables."""
        agents = {}

        # Look for AGENT_*_USER_ID environment variables
        for key, value in os.environ.items():
            if key.startswith("AGENT_") and key.endswith("_USER_ID"):
                # Extract agent name: AGENT_VEGA_USER_ID -> vega
                agent_name = key[6:-8].lower()  # Remove "AGENT_" and "_USER_ID"
                try:
                    agents[agent_name] = int(value)
                except ValueError:
                    pass

        return cls(agents=agents)

    def get_user_id(self, agent_name: str) -> Optional[int]:
        """Get Discord user ID for an agent."""
        return self.agents.get(agent_name.lower())

    def get_agent_name(self, user_id: int) -> Optional[str]:
        """Get agent name from Discord user ID."""
        for name, uid in self.agents.items():
            if uid == user_id:
                return name
        return None


@dataclass
class BaseAgentConfig:
    """Base configuration for any agent bot."""
    # Agent identity
    agent_name: str

    # Discord
    discord_token: str
    bot_user_id: int

    # Security
    allowed_user_id: int

    # LLM (primary - for main agent reasoning)
    llm_provider: str = "gemini"
    llm_api_key: str = ""
    llm_model: Optional[str] = None

    # Utility LLM (for quick evaluations like chime-in - use cheaper/faster model)
    utility_llm_provider: Optional[str] = None  # Falls back to llm_provider if not set
    utility_llm_api_key: Optional[str] = None   # Falls back to llm_api_key if not set
    utility_llm_model: Optional[str] = None     # Falls back to llm_model if not set

    # Thinking LLM (for complex reasoning, architecture decisions - use reasoning model)
    thinking_llm_provider: Optional[str] = None  # Falls back to llm_provider if not set
    thinking_llm_api_key: Optional[str] = None   # Falls back to llm_api_key if not set
    thinking_llm_model: Optional[str] = None     # Falls back to llm_model if not set

    # Ollama-specific
    ollama_base_url: Optional[str] = None  # Default: http://localhost:11434

    # Database
    redis_url: Optional[str] = None

    # Workspace
    workspace_dir: str = "/tmp/workspace"

    # Agent registry (for @mentions)
    agent_registry: AgentRegistryConfig = field(default_factory=AgentRegistryConfig)

    # Main channel for collaborative agent communication and events
    main_channel_id: Optional[int] = None

    @classmethod
    def from_env(cls, agent_name: str) -> 'BaseAgentConfig':
        """
        Load configuration from environment variables.

        Uses agent-specific variables with fallback to global ones.
        E.g., ALTAIR_LLM_MODEL -> LLM_MODEL
        """
        prefix = agent_name.upper()

        def get_env(key: str, default: str = "") -> str:
            # Try agent-specific first, then global
            return os.getenv(f"{prefix}_{key}") or os.getenv(key, default)

        # Use AGENT_CHANNEL_ID for main collaborative channel
        agent_channel = os.getenv("AGENT_CHANNEL_ID")

        # Utility LLM settings (for cheap/fast evaluations like chime-in)
        # Falls back to primary LLM if not set
        utility_provider = get_env("UTILITY_LLM_PROVIDER") or None
        utility_api_key = get_env("UTILITY_LLM_API_KEY") or None
        utility_model = get_env("UTILITY_LLM_MODEL") or None

        # Thinking LLM settings (for complex reasoning)
        # Falls back to primary LLM if not set
        thinking_provider = get_env("THINKING_LLM_PROVIDER") or None
        thinking_api_key = get_env("THINKING_LLM_API_KEY") or None
        thinking_model = get_env("THINKING_LLM_MODEL") or None

        return cls(
            agent_name=agent_name,
            discord_token=get_env("DISCORD_TOKEN"),
            bot_user_id=int(get_env("BOT_USER_ID", "0")),
            allowed_user_id=int(get_env("ALLOWED_USER_ID", "0")),
            llm_provider=get_env("LLM_PROVIDER", "gemini"),
            llm_api_key=get_env("LLM_API_KEY"),
            llm_model=get_env("LLM_MODEL") or None,
            utility_llm_provider=utility_provider,
            utility_llm_api_key=utility_api_key,
            utility_llm_model=utility_model,
            thinking_llm_provider=thinking_provider,
            thinking_llm_api_key=thinking_api_key,
            thinking_llm_model=thinking_model,
            ollama_base_url=get_env("OLLAMA_BASE_URL") or None,
            redis_url=get_env("REDIS_URL") or None,
            workspace_dir=get_env("WORKSPACE_DIR", "/tmp/workspace"),
            agent_registry=AgentRegistryConfig.from_env(),
            main_channel_id=int(agent_channel) if agent_channel else None,
        )


@dataclass
class SharedConfig:
    """
    Shared configuration accessible to all agents.

    This is for global settings that don't belong to a specific agent.
    """
    # Database
    redis_url: Optional[str] = None

    # Workspace
    workspace_dir: str = "/tmp/workspace"

    # Agent registry
    agent_registry: AgentRegistryConfig = field(default_factory=AgentRegistryConfig)

    # Main channel for collaborative agent communication and events
    main_channel_id: Optional[int] = None

    @classmethod
    def from_env(cls) -> 'SharedConfig':
        """Load shared configuration from environment."""
        agent_channel = os.getenv("AGENT_CHANNEL_ID")
        return cls(
            redis_url=os.getenv("REDIS_URL"),
            workspace_dir=os.getenv("WORKSPACE_DIR", "/tmp/workspace"),
            agent_registry=AgentRegistryConfig.from_env(),
            main_channel_id=int(agent_channel) if agent_channel else None,
        )
