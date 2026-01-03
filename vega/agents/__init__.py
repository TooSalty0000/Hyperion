"""Vega agent system.

Vega is the head conversational agent. For hands-on work,
Vega communicates with Altair (a separate Discord bot) via @mentions.
"""

# Re-export from shared for convenience
from shared.base_agent import BaseAgent, AgentContext, AgentResponse

# Vega-specific
from .vega import VegaAgent

__all__ = [
    "BaseAgent",
    "AgentContext",
    "AgentResponse",
    "VegaAgent",
]
