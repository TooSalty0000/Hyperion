"""Soul data models for agent personality persistence."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class TraitType(Enum):
    """Types of soul traits."""
    CORE_VALUE = "core_value"           # Fundamental beliefs, rarely change
    BEHAVIORAL = "behavioral"            # How the agent acts
    PREFERENCE = "preference"            # Contextual adaptations (learned preferences)
    SKILL = "skill"                      # Self-assessment of capabilities
    RELATIONSHIP = "relationship"        # Understanding of users/other agents


class TraitStability(Enum):
    """
    Stability levels for traits - higher stability = slower change.

    Evolution path: NOVICE → DEVELOPING → ESTABLISHED → CORE
    """
    NOVICE = "novice"           # New trait, easily changed (0-3 reinforcements)
    DEVELOPING = "developing"   # Growing trait (3-10 reinforcements)
    ESTABLISHED = "established" # Stable trait (10-50 reinforcements)
    CORE = "core"              # Fundamental, very hard to change (50+ reinforcements)


# Configuration constants
REINFORCEMENT_DELTA = 0.05          # Strength increase per reinforcement
CONTRADICTION_DELTA = 0.08          # Strength decrease per contradiction
DAILY_DECAY_RATE = 0.02             # 2% decay per day for inactive traits
MIN_STRENGTH_THRESHOLD = 0.1        # Below this, trait is considered dormant
MAX_STRENGTH = 1.0
MIN_STRENGTH = 0.0

# Stability thresholds (reinforcement counts)
DEVELOPING_THRESHOLD = 3
ESTABLISHED_THRESHOLD = 10
CORE_THRESHOLD = 50

# Stability thresholds as dict (for tests and lookup)
STABILITY_THRESHOLDS = {
    TraitStability.NOVICE: DEVELOPING_THRESHOLD,
    TraitStability.DEVELOPING: ESTABLISHED_THRESHOLD,
    TraitStability.ESTABLISHED: CORE_THRESHOLD,
}

# Stability resistance multipliers (higher = harder to change)
STABILITY_RESISTANCE = {
    TraitStability.NOVICE: 1.0,
    TraitStability.DEVELOPING: 0.8,
    TraitStability.ESTABLISHED: 0.5,
    TraitStability.CORE: 0.2,
}


@dataclass
class TraitEvolution:
    """
    Records a single evolution event for a trait.

    Tracks how and why a trait changed over time.
    """
    id: str                                     # UUID
    trait_id: str                               # Parent trait ID
    agent_id: str                               # Agent this belongs to
    event_type: str                             # "reinforce", "contradict", "decay", "resolve"
    trigger: str                                # What triggered this evolution
    evidence: Optional[str] = None              # Supporting evidence for the change
    strength_before: float = 0.0
    strength_after: float = 0.0
    stability_before: Optional[str] = None      # Stability level before (as string)
    stability_after: Optional[str] = None       # Stability level after (as string)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for Redis storage."""
        return {
            "id": self.id,
            "trait_id": self.trait_id,
            "agent_id": self.agent_id,
            "event_type": self.event_type,
            "trigger": self.trigger,
            "evidence": self.evidence,
            "strength_before": self.strength_before,
            "strength_after": self.strength_after,
            "stability_before": self.stability_before,
            "stability_after": self.stability_after,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraitEvolution':
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            trait_id=data["trait_id"],
            agent_id=data["agent_id"],
            event_type=data["event_type"],
            trigger=data["trigger"],
            evidence=data.get("evidence"),
            strength_before=data.get("strength_before", 0.0),
            strength_after=data.get("strength_after", 0.0),
            stability_before=data.get("stability_before"),
            stability_after=data.get("stability_after"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
        )


@dataclass
class SoulTrait:
    """
    A single personality trait that defines who the agent is.

    Unlike memories (facts), traits represent values, tendencies, and self-identity.
    They evolve slowly through reinforcement, contradiction, and reflection.
    """
    id: str                                     # UUID
    agent_id: str                               # "vega", "altair", "polaris", "canopus"
    trait_type: TraitType                       # Type of trait
    name: str                                   # Short identifier (e.g., "genuine_helpfulness")
    description: str                            # What this trait means
    expression: str                             # How it manifests in behavior

    # Evolution state
    strength: float = 0.5                       # 0.0-1.0 how strong this trait is
    stability: TraitStability = TraitStability.NOVICE
    confidence: float = 0.5                     # 0.0-1.0 self-assessed confidence

    # Tracking
    reinforcement_count: int = 0               # Times reinforced
    contradiction_count: int = 0               # Times contradicted
    last_activated: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Context
    context_tags: List[str] = field(default_factory=list)  # When this applies
    evolution_history: List[str] = field(default_factory=list)  # TraitEvolution IDs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for Redis storage."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "trait_type": self.trait_type.value,
            "name": self.name,
            "description": self.description,
            "expression": self.expression,
            "strength": self.strength,
            "stability": self.stability.value,
            "confidence": self.confidence,
            "reinforcement_count": self.reinforcement_count,
            "contradiction_count": self.contradiction_count,
            "last_activated": self.last_activated.isoformat(),
            "created_at": self.created_at.isoformat(),
            "context_tags": self.context_tags,
            "evolution_history": self.evolution_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SoulTrait':
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            trait_type=TraitType(data.get("trait_type", "core_value")),
            name=data["name"],
            description=data["description"],
            expression=data["expression"],
            strength=data.get("strength", 0.5),
            stability=TraitStability(data.get("stability", "novice")),
            confidence=data.get("confidence", 0.5),
            reinforcement_count=data.get("reinforcement_count", 0),
            contradiction_count=data.get("contradiction_count", 0),
            last_activated=datetime.fromisoformat(data["last_activated"]) if data.get("last_activated") else datetime.utcnow(),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            context_tags=data.get("context_tags", []),
            evolution_history=data.get("evolution_history", []),
        )

    def format_for_prompt(self) -> str:
        """Format trait for inclusion in system prompt."""
        strength_indicator = self._strength_indicator()
        return f"- [{strength_indicator}] {self.expression}"

    def _strength_indicator(self) -> str:
        """Get visual indicator of trait strength."""
        if self.strength >= 0.8:
            return "STRONG"
        elif self.strength >= 0.5:
            return "moderate"
        elif self.strength >= 0.3:
            return "weak"
        else:
            return "fading"

    def get_stability_resistance(self) -> float:
        """Get the resistance multiplier based on stability level."""
        return STABILITY_RESISTANCE.get(self.stability, 1.0)

    def should_advance_stability(self) -> bool:
        """Check if trait should advance to next stability level."""
        if self.stability == TraitStability.NOVICE:
            return self.reinforcement_count >= DEVELOPING_THRESHOLD
        elif self.stability == TraitStability.DEVELOPING:
            return self.reinforcement_count >= ESTABLISHED_THRESHOLD
        elif self.stability == TraitStability.ESTABLISHED:
            return self.reinforcement_count >= CORE_THRESHOLD
        return False  # CORE doesn't advance

    def get_next_stability(self) -> TraitStability:
        """Get the next stability level."""
        if self.stability == TraitStability.NOVICE:
            return TraitStability.DEVELOPING
        elif self.stability == TraitStability.DEVELOPING:
            return TraitStability.ESTABLISHED
        elif self.stability == TraitStability.ESTABLISHED:
            return TraitStability.CORE
        return TraitStability.CORE

    def advance_stability(self) -> bool:
        """
        Advance to next stability level if threshold is met.

        Returns True if stability was advanced.
        """
        if self.should_advance_stability():
            self.stability = self.get_next_stability()
            return True
        return False


@dataclass
class SoulContext:
    """
    Context provided to agent prompts about their personality/soul.

    Built by SoulManager.build_soul_context() and formatted for prompt injection.
    Soul context appears BEFORE memory context in the prompt.
    """
    core_values: List[SoulTrait] = field(default_factory=list)
    behavioral_traits: List[SoulTrait] = field(default_factory=list)
    learned_preferences: List[SoulTrait] = field(default_factory=list)
    skill_confidences: List[SoulTrait] = field(default_factory=list)
    relationships: List[SoulTrait] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """
        Format complete soul context as prompt text.

        This is injected into the system prompt BEFORE memories.
        """
        sections = []

        # Core values - fundamental beliefs
        if self.core_values:
            lines = ["### CORE VALUES (Who You Are):"]
            for trait in sorted(self.core_values, key=lambda t: t.strength, reverse=True):
                lines.append(trait.format_for_prompt())
            sections.append("\n".join(lines))

        # Behavioral traits - how you act
        if self.behavioral_traits:
            lines = ["### BEHAVIORAL TENDENCIES:"]
            for trait in sorted(self.behavioral_traits, key=lambda t: t.strength, reverse=True):
                lines.append(trait.format_for_prompt())
            sections.append("\n".join(lines))

        # Learned preferences - contextual adaptations
        if self.learned_preferences:
            lines = ["### LEARNED PREFERENCES:"]
            for trait in sorted(self.learned_preferences, key=lambda t: t.strength, reverse=True):
                lines.append(trait.format_for_prompt())
            sections.append("\n".join(lines))

        # Skill confidences - self-assessment
        if self.skill_confidences:
            lines = ["### SKILL CONFIDENCE:"]
            for trait in sorted(self.skill_confidences, key=lambda t: t.strength, reverse=True):
                conf_level = self._confidence_level(trait.confidence)
                lines.append(f"- {trait.name}: {conf_level} confidence - {trait.expression}")
            sections.append("\n".join(lines))

        # Relationships - understanding of others
        if self.relationships:
            lines = ["### RELATIONSHIPS:"]
            for trait in sorted(self.relationships, key=lambda t: t.strength, reverse=True):
                lines.append(trait.format_for_prompt())
            sections.append("\n".join(lines))

        if not sections:
            return "Your soul is still forming. Act authentically and let your personality develop naturally."

        return "\n\n".join(sections)

    def _confidence_level(self, confidence: float) -> str:
        """Convert confidence float to descriptive level."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "moderate"
        elif confidence >= 0.3:
            return "developing"
        else:
            return "low"

    def has_traits(self) -> bool:
        """Check if there are any traits in the context."""
        return bool(
            self.core_values or
            self.behavioral_traits or
            self.learned_preferences or
            self.skill_confidences or
            self.relationships
        )

    def get_all_traits(self) -> List[SoulTrait]:
        """Get all traits as a flat list."""
        return (
            self.core_values +
            self.behavioral_traits +
            self.learned_preferences +
            self.skill_confidences +
            self.relationships
        )
