"""Shared module for multi-agent Discord bot system.

This module contains common abstractions used by all agents:
- BaseAgent, AgentContext, AgentResponse
- LLM providers
- Session management
- Tool interfaces
- Database abstractions
- Message queue for sequential processing
"""

from .base_agent import BaseAgent, AgentContext, AgentResponse
from .agent_queue import AgentMessageQueue, QueuedMessage, QueueItemStatus

__all__ = [
    "BaseAgent",
    "AgentContext",
    "AgentResponse",
    "AgentMessageQueue",
    "QueuedMessage",
    "QueueItemStatus",
]
