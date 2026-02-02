"""Tests for the Soul System - personality trait management for agents."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from shared.soul.models import (
    SoulTrait,
    TraitEvolution,
    SoulContext,
    TraitType,
    TraitStability,
    REINFORCEMENT_DELTA,
    CONTRADICTION_DELTA,
    DAILY_DECAY_RATE,
    STABILITY_THRESHOLDS,
    STABILITY_RESISTANCE,
)
from shared.soul.defaults import get_default_traits, AGENT_DEFAULT_TRAITS


class TestSoulTrait:
    """Tests for SoulTrait model."""

    def test_trait_defaults(self):
        trait = SoulTrait(
            id="test-trait",
            agent_id="vega",
            trait_type=TraitType.CORE_VALUE,
            name="test_value",
            description="A test core value",
            expression="Expresses as testing behavior",
        )
        assert trait.strength == 0.5
        assert trait.stability == TraitStability.NOVICE
        assert trait.confidence == 0.5
        assert trait.reinforcement_count == 0
        assert trait.contradiction_count == 0
        assert trait.context_tags == []
        assert trait.evolution_history == []

    def test_trait_from_dict_full(self):
        data = {
            "id": "t1",
            "agent_id": "altair",
            "trait_type": "behavioral",
            "name": "thoroughness",
            "description": "Checks work carefully",
            "expression": "Verifies before reporting",
            "strength": 0.8,
            "stability": "established",
            "confidence": 0.9,
            "reinforcement_count": 25,
            "contradiction_count": 2,
            "last_activated": "2024-01-15T12:00:00+00:00",
            "created_at": "2024-01-01T00:00:00+00:00",
            "context_tags": ["cli", "code"],
            "evolution_history": ["evo-1", "evo-2"],
        }
        trait = SoulTrait.from_dict(data)

        assert trait.id == "t1"
        assert trait.agent_id == "altair"
        assert trait.trait_type == TraitType.BEHAVIORAL
        assert trait.name == "thoroughness"
        assert trait.strength == 0.8
        assert trait.stability == TraitStability.ESTABLISHED
        assert trait.confidence == 0.9
        assert trait.reinforcement_count == 25
        assert trait.context_tags == ["cli", "code"]

    def test_trait_from_dict_minimal(self):
        data = {
            "id": "t2",
            "agent_id": "polaris",
            "trait_type": "preference",
            "name": "brevity",
            "description": "Prefers brief responses",
            "expression": "Uses concise language",
        }
        trait = SoulTrait.from_dict(data)

        assert trait.strength == 0.5
        assert trait.stability == TraitStability.NOVICE
        assert trait.created_at is not None

    def test_trait_to_dict(self):
        trait = SoulTrait(
            id="t3",
            agent_id="canopus",
            trait_type=TraitType.SKILL,
            name="web_navigation",
            description="Navigation skill",
            expression="Efficiently navigates pages",
            strength=0.7,
            stability=TraitStability.DEVELOPING,
        )
        data = trait.to_dict()

        assert data["id"] == "t3"
        assert data["trait_type"] == "skill"
        assert data["stability"] == "developing"
        assert data["strength"] == 0.7
        assert "created_at" in data

    def test_trait_roundtrip(self):
        """Serialize and deserialize should preserve all fields."""
        original = SoulTrait(
            id="round-trip",
            agent_id="vega",
            trait_type=TraitType.RELATIONSHIP,
            name="user_trust",
            description="Trust relationship with user",
            expression="Shows appropriate confidence",
            strength=0.6,
            stability=TraitStability.CORE,
            confidence=0.85,
            reinforcement_count=100,
            contradiction_count=5,
            context_tags=["user", "trust"],
            evolution_history=["e1", "e2"],
        )
        data = original.to_dict()
        restored = SoulTrait.from_dict(data)

        assert restored.id == original.id
        assert restored.trait_type == original.trait_type
        assert restored.strength == original.strength
        assert restored.stability == original.stability
        assert restored.reinforcement_count == original.reinforcement_count
        assert restored.context_tags == original.context_tags

    def test_format_for_prompt(self):
        trait = SoulTrait(
            id="t4",
            agent_id="altair",
            trait_type=TraitType.CORE_VALUE,
            name="craftsmanship",
            description="Values quality work",
            expression="Takes time to do things right",
            strength=0.9,
            stability=TraitStability.ESTABLISHED,
        )
        formatted = trait.format_for_prompt()

        # format_for_prompt returns "- [STRONG] Takes time to do things right"
        assert "Takes time to do things right" in formatted
        assert "STRONG" in formatted  # strength >= 0.8 = STRONG

    def test_should_advance_stability(self):
        trait = SoulTrait(
            id="t5",
            agent_id="vega",
            trait_type=TraitType.BEHAVIORAL,
            name="test",
            description="test",
            expression="test",
            stability=TraitStability.NOVICE,
            reinforcement_count=2,
        )
        assert not trait.should_advance_stability()

        trait.reinforcement_count = 3
        assert trait.should_advance_stability()

    def test_advance_stability_progression(self):
        trait = SoulTrait(
            id="t6",
            agent_id="vega",
            trait_type=TraitType.BEHAVIORAL,
            name="test",
            description="test",
            expression="test",
            stability=TraitStability.NOVICE,
        )

        # NOVICE -> DEVELOPING at 3 reinforcements
        trait.reinforcement_count = 3
        trait.advance_stability()
        assert trait.stability == TraitStability.DEVELOPING

        # DEVELOPING -> ESTABLISHED at 10 reinforcements
        trait.reinforcement_count = 10
        trait.advance_stability()
        assert trait.stability == TraitStability.ESTABLISHED

        # ESTABLISHED -> CORE at 50 reinforcements
        trait.reinforcement_count = 50
        trait.advance_stability()
        assert trait.stability == TraitStability.CORE

        # CORE stays CORE
        trait.reinforcement_count = 100
        old_stability = trait.stability
        trait.advance_stability()
        assert trait.stability == old_stability


class TestTraitEvolution:
    """Tests for TraitEvolution event tracking."""

    def test_evolution_defaults(self):
        evo = TraitEvolution(
            id="evo-1",
            trait_id="t1",
            agent_id="vega",
            event_type="reinforcement",
            trigger="User praised work",
        )
        assert evo.strength_before == 0.0
        assert evo.strength_after == 0.0
        assert evo.created_at is not None

    def test_evolution_from_dict(self):
        data = {
            "id": "evo-2",
            "trait_id": "t2",
            "agent_id": "altair",
            "event_type": "contradiction",
            "trigger": "Task failed",
            "evidence": "Build error occurred",
            "strength_before": 0.7,
            "strength_after": 0.62,
            "created_at": "2024-01-15T12:00:00",
        }
        evo = TraitEvolution.from_dict(data)

        assert evo.id == "evo-2"
        assert evo.event_type == "contradiction"
        assert evo.strength_before == 0.7
        assert evo.strength_after == 0.62

    def test_evolution_to_dict(self):
        evo = TraitEvolution(
            id="evo-3",
            trait_id="t3",
            agent_id="polaris",
            event_type="decay",
            trigger="Time passed",
            strength_before=0.5,
            strength_after=0.49,
        )
        data = evo.to_dict()

        assert data["event_type"] == "decay"
        assert data["strength_before"] == 0.5


class TestSoulContext:
    """Tests for SoulContext prompt formatting."""

    def test_empty_context(self):
        ctx = SoulContext()
        assert not ctx.has_traits()
        formatted = ctx.format_for_prompt()
        # Empty context returns a default message about personality forming
        assert "forming" in formatted or "develop" in formatted

    def test_context_with_traits(self):
        core_value = SoulTrait(
            id="cv1",
            agent_id="vega",
            trait_type=TraitType.CORE_VALUE,
            name="helpfulness",
            description="Values being helpful",
            expression="Always tries to assist",
            strength=0.9,
            stability=TraitStability.CORE,
        )
        behavioral = SoulTrait(
            id="bh1",
            agent_id="vega",
            trait_type=TraitType.BEHAVIORAL,
            name="verification",
            description="Verifies work",
            expression="Double-checks results",
            strength=0.7,
            stability=TraitStability.ESTABLISHED,
        )

        ctx = SoulContext(
            core_values=[core_value],
            behavioral_traits=[behavioral],
        )

        assert ctx.has_traits()
        formatted = ctx.format_for_prompt()

        assert "CORE VALUES" in formatted
        assert "Always tries to assist" in formatted  # Expression is in output
        assert "BEHAVIORAL" in formatted
        assert "Double-checks results" in formatted  # Expression is in output

    def test_context_all_trait_types(self):
        ctx = SoulContext(
            core_values=[
                SoulTrait(
                    id="cv", agent_id="a", trait_type=TraitType.CORE_VALUE,
                    name="value", description="d", expression="e"
                )
            ],
            behavioral_traits=[
                SoulTrait(
                    id="bt", agent_id="a", trait_type=TraitType.BEHAVIORAL,
                    name="behavior", description="d", expression="e"
                )
            ],
            learned_preferences=[
                SoulTrait(
                    id="lp", agent_id="a", trait_type=TraitType.PREFERENCE,
                    name="preference", description="d", expression="e"
                )
            ],
            skill_confidences=[
                SoulTrait(
                    id="sc", agent_id="a", trait_type=TraitType.SKILL,
                    name="skill", description="d", expression="skill_expression"
                )
            ],
            relationships=[
                SoulTrait(
                    id="rl", agent_id="a", trait_type=TraitType.RELATIONSHIP,
                    name="relation", description="d", expression="e"
                )
            ],
        )

        formatted = ctx.format_for_prompt()
        assert "CORE VALUES" in formatted
        assert "BEHAVIORAL" in formatted
        assert "LEARNED PREFERENCES" in formatted
        assert "SKILL CONFIDENCE" in formatted
        assert "RELATIONSHIPS" in formatted


class TestEvolutionMechanics:
    """Tests for reinforcement, decay, and stability resistance."""

    def test_stability_thresholds(self):
        """Verify stability advancement thresholds."""
        assert STABILITY_THRESHOLDS[TraitStability.NOVICE] == 3
        assert STABILITY_THRESHOLDS[TraitStability.DEVELOPING] == 10
        assert STABILITY_THRESHOLDS[TraitStability.ESTABLISHED] == 50

    def test_stability_resistance(self):
        """Higher stability should have lower multiplier (= more resistance to change).

        Resistance works as a multiplier on change deltas:
        - NOVICE (1.0): full change applied
        - CORE (0.2): only 20% of change applied
        """
        # Higher stability means lower multiplier (more resistant to change)
        assert STABILITY_RESISTANCE[TraitStability.NOVICE] > STABILITY_RESISTANCE[TraitStability.DEVELOPING]
        assert STABILITY_RESISTANCE[TraitStability.DEVELOPING] > STABILITY_RESISTANCE[TraitStability.ESTABLISHED]
        assert STABILITY_RESISTANCE[TraitStability.ESTABLISHED] > STABILITY_RESISTANCE[TraitStability.CORE]
        # CORE should be the most resistant (lowest multiplier)
        assert STABILITY_RESISTANCE[TraitStability.CORE] == 0.2

    def test_reinforcement_delta(self):
        """Base reinforcement should increase strength."""
        assert REINFORCEMENT_DELTA > 0
        assert REINFORCEMENT_DELTA == 0.05

    def test_contradiction_delta(self):
        """Contradiction should decrease strength more than reinforcement increases it."""
        assert CONTRADICTION_DELTA > 0
        assert CONTRADICTION_DELTA > REINFORCEMENT_DELTA
        assert CONTRADICTION_DELTA == 0.08

    def test_decay_rate(self):
        """Decay should be small but present."""
        assert DAILY_DECAY_RATE > 0
        assert DAILY_DECAY_RATE < 0.1


class TestDefaultTraits:
    """Tests for default trait definitions."""

    def test_all_agents_have_defaults(self):
        """Each agent should have at least one default trait."""
        for agent in ["vega", "altair", "polaris", "canopus"]:
            traits = get_default_traits(agent)
            assert len(traits) >= 1, f"{agent} should have at least one default trait"

    def test_unknown_agent_returns_empty(self):
        """Unknown agents should return empty list."""
        traits = get_default_traits("unknown_agent")
        assert traits == []

    def test_vega_default_traits(self):
        """Vega should have genuine helpfulness core value."""
        traits = get_default_traits("vega")
        assert len(traits) == 1
        trait = traits[0]
        assert trait["trait_type"] == "core_value"
        assert "helpfulness" in trait["name"].lower() or "genuine" in trait["name"].lower()

    def test_altair_default_traits(self):
        """Altair should have craftsmanship core value."""
        traits = get_default_traits("altair")
        assert len(traits) == 1
        trait = traits[0]
        assert trait["trait_type"] == "core_value"
        assert "craftsmanship" in trait["name"].lower()

    def test_polaris_default_traits(self):
        """Polaris should have time respect core value."""
        traits = get_default_traits("polaris")
        assert len(traits) == 1
        trait = traits[0]
        assert trait["trait_type"] == "core_value"
        assert "time" in trait["name"].lower()

    def test_canopus_default_traits(self):
        """Canopus should have evidence-based core value."""
        traits = get_default_traits("canopus")
        assert len(traits) == 1
        trait = traits[0]
        assert trait["trait_type"] == "core_value"
        assert "evidence" in trait["name"].lower()

    def test_default_traits_structure(self):
        """All default traits should have valid structure."""
        for agent_id, traits in AGENT_DEFAULT_TRAITS.items():
            for trait in traits:
                assert trait.get("trait_type") is not None
                assert trait.get("name") is not None
                assert trait.get("description") is not None
                assert trait.get("expression") is not None
                assert trait.get("stability") == "novice"
                assert 0.0 <= trait.get("strength", 0.5) <= 1.0


class TestTraitType:
    """Tests for TraitType enum."""

    def test_trait_type_values(self):
        assert TraitType.CORE_VALUE.value == "core_value"
        assert TraitType.BEHAVIORAL.value == "behavioral"
        assert TraitType.PREFERENCE.value == "preference"
        assert TraitType.SKILL.value == "skill"
        assert TraitType.RELATIONSHIP.value == "relationship"


class TestTraitStability:
    """Tests for TraitStability enum."""

    def test_stability_values(self):
        assert TraitStability.NOVICE.value == "novice"
        assert TraitStability.DEVELOPING.value == "developing"
        assert TraitStability.ESTABLISHED.value == "established"
        assert TraitStability.CORE.value == "core"

    def test_stability_ordering(self):
        """Stability levels should be ordered from least to most stable."""
        levels = list(TraitStability)
        assert levels == [
            TraitStability.NOVICE,
            TraitStability.DEVELOPING,
            TraitStability.ESTABLISHED,
            TraitStability.CORE,
        ]
