"""Altair - The autonomous project manager Discord bot.

Altair is a separate Discord bot that:
- Manages terminal sessions and CLI agents (Claude Code, etc.)
- Executes coding tasks autonomously
- Asks for user permission before sensitive actions
- Communicates with other agents via @mentions
"""

from .agent import AltairAgent

__all__ = ["AltairAgent"]
