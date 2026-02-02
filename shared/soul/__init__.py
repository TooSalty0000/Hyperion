"""Soul system for agent personality persistence.

The soul system provides personality traits that evolve organically over time,
separate from factual memories. Each agent develops their own "soul" through
reinforcement, decay, and contradiction resolution - mimicking human personality
development.

Key differences from memories:
- Soul = WHO the agent is (values, traits, tendencies)
- Memory = WHAT the agent knows (facts, context, history)
- Soul changes gradually through reinforcement
- Memory updates frequently based on new information
- Soul is injected first in the prompt, before memories
"""

from .models import (
    SoulTrait,
    SoulContext,
    TraitEvolution,
    TraitType,
    TraitStability,
    REINFORCEMENT_DELTA,
    CONTRADICTION_DELTA,
    DAILY_DECAY_RATE,
    MIN_STRENGTH_THRESHOLD,
)
from .manager import SoulManager
from .defaults import get_default_traits, get_all_default_traits
from .tools import get_soul_tools

__all__ = [
    # Core classes
    "SoulManager",
    "SoulTrait",
    "SoulContext",
    "TraitEvolution",
    # Enums
    "TraitType",
    "TraitStability",
    # Constants
    "REINFORCEMENT_DELTA",
    "CONTRADICTION_DELTA",
    "DAILY_DECAY_RATE",
    "MIN_STRENGTH_THRESHOLD",
    # Functions
    "get_default_traits",
    "get_all_default_traits",
    "get_soul_tools",
]
