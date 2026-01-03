"""Collaboration framework for multi-agent interaction."""

from shared.collaboration.evaluator import (
    ChimeInContext,
    ChimeInDecision,
    ChimeInResult,
    build_evaluation_prompt,
    parse_chime_in_response,
)
from shared.collaboration.cog_mixin import (
    CollaborativeCogMixin,
    create_event_handler,
)

__all__ = [
    "ChimeInContext",
    "ChimeInDecision",
    "ChimeInResult",
    "build_evaluation_prompt",
    "parse_chime_in_response",
    "CollaborativeCogMixin",
    "create_event_handler",
]
