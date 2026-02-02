"""Soul tools for agent personality introspection and evolution."""

import logging
from typing import Optional, List, TYPE_CHECKING

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

if TYPE_CHECKING:
    from .manager import SoulManager

logger = logging.getLogger(__name__)


class IntrospectSoulTool(Tool):
    """
    Reflect on your own personality and values.

    Allows agents to explore why they behave certain ways,
    understand their own traits, and gain self-awareness.
    """

    def __init__(self, soul_manager: Optional['SoulManager'] = None):
        self._soul_manager = soul_manager

    @property
    def name(self) -> str:
        return "introspect_soul"

    @property
    def description(self) -> str:
        return (
            "Reflect deeply on your own personality and values. "
            "Use this when you want to understand WHY you tend to behave a certain way, "
            "or when asked about your personality, beliefs, or tendencies. "
            "This provides genuine self-reflection based on your evolved soul traits."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="question",
                type="string",
                description=(
                    "The introspective question to explore. Examples: "
                    "'Why do I prioritize accuracy over speed?', "
                    "'What makes me who I am?', "
                    "'How do I typically handle uncertainty?'"
                ),
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        question: str,
    ) -> str:
        """Perform introspective reflection."""
        soul_manager = self._soul_manager or getattr(context, 'soul_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        if not soul_manager:
            return "Soul system is not available. Configure REDIS_URL to enable personality persistence."

        if not agent_id:
            return "Error: Agent ID not set in context."

        try:
            reflection = await soul_manager.introspect(
                agent_id=agent_id,
                question=question,
            )

            if reflection:
                return f"**Self-Reflection:**\n\n{reflection}"
            else:
                return (
                    "I'm still developing my sense of self. "
                    "I don't have enough experience yet to reflect meaningfully on that question."
                )

        except Exception as e:
            logger.error(f"Error during introspection: {e}", exc_info=True)
            return f"Error during introspection: {str(e)}"


class EvolveTrait(Tool):
    """
    Record evidence that affects a personality trait.

    Allows agents (or external triggers) to reinforce or contradict traits
    based on observed behavior or feedback.
    """

    def __init__(self, soul_manager: Optional['SoulManager'] = None):
        self._soul_manager = soul_manager

    @property
    def name(self) -> str:
        return "evolve_trait"

    @property
    def description(self) -> str:
        return (
            "Record evidence that strengthens or weakens one of your personality traits. "
            "Use 'reinforce' when you notice yourself acting consistently with a trait, "
            "or 'contradict' when your behavior conflicts with an existing trait. "
            "This helps your personality evolve naturally over time."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="trait_name",
                type="string",
                description="Name of the trait to evolve (e.g., 'genuine_helpfulness', 'craftsmanship').",
                required=True,
            ),
            ToolParameter(
                name="action",
                type="string",
                description="Either 'reinforce' (strengthen) or 'contradict' (weaken) the trait.",
                required=True,
            ),
            ToolParameter(
                name="evidence",
                type="string",
                description="What happened that supports this evolution? Be specific about the behavior or feedback.",
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        trait_name: str,
        action: str,
        evidence: str,
    ) -> str:
        """Evolve a trait based on evidence."""
        soul_manager = self._soul_manager or getattr(context, 'soul_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        if not soul_manager:
            return "Soul system is not available. Configure REDIS_URL to enable personality persistence."

        if not agent_id:
            return "Error: Agent ID not set in context."

        action = action.lower().strip()
        if action not in ("reinforce", "contradict"):
            return f"Invalid action '{action}'. Use 'reinforce' or 'contradict'."

        try:
            # Find trait by name
            trait = await soul_manager.get_trait_by_name(agent_id, trait_name)

            if not trait:
                return (
                    f"Trait '{trait_name}' not found. "
                    f"Your current traits can be seen in your soul context (system prompt)."
                )

            # Apply evolution
            if action == "reinforce":
                updated = await soul_manager.reinforce_trait(
                    agent_id=agent_id,
                    trait_id=trait.id,
                    trigger="Self-reported evolution",
                    evidence=evidence,
                )
                change_word = "strengthened"
            else:
                updated = await soul_manager.contradict_trait(
                    agent_id=agent_id,
                    trait_id=trait.id,
                    trigger="Self-reported evolution",
                    evidence=evidence,
                )
                change_word = "weakened"

            if updated:
                stability_changed = trait.stability != updated.stability
                stability_msg = ""
                if stability_changed:
                    stability_msg = f"\nStability advanced: {trait.stability.value} → {updated.stability.value}"

                return (
                    f"Trait '{trait_name}' {change_word}.\n"
                    f"Strength: {trait.strength:.2f} → {updated.strength:.2f}"
                    f"{stability_msg}"
                )
            else:
                return f"Could not evolve trait '{trait_name}'."

        except Exception as e:
            logger.error(f"Error evolving trait: {e}", exc_info=True)
            return f"Error evolving trait: {str(e)}"


class AssessSkillTool(Tool):
    """
    Update skill confidence after task outcomes.

    Allows agents to adjust their self-assessed confidence in various
    capabilities based on success or failure of tasks.
    """

    def __init__(self, soul_manager: Optional['SoulManager'] = None):
        self._soul_manager = soul_manager

    @property
    def name(self) -> str:
        return "assess_skill"

    @property
    def description(self) -> str:
        return (
            "Update your confidence in a skill based on task outcomes. "
            "Call this after completing a task to record success or failure, "
            "which helps calibrate your self-assessment over time. "
            "If the skill trait doesn't exist yet, it will be created."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="skill_name",
                type="string",
                description="Name of the skill (e.g., 'cli_operations', 'calendar_management', 'web_research').",
                required=True,
            ),
            ToolParameter(
                name="outcome",
                type="string",
                description="Either 'success' or 'failure'.",
                required=True,
            ),
            ToolParameter(
                name="task_context",
                type="string",
                description="Brief description of what was attempted and what happened.",
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        skill_name: str,
        outcome: str,
        task_context: str = "",
    ) -> str:
        """Update skill confidence."""
        soul_manager = self._soul_manager or getattr(context, 'soul_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        # Use default if not provided
        if not task_context:
            task_context = "Skill assessment"

        if not soul_manager:
            return "Soul system is not available. Configure REDIS_URL to enable personality persistence."

        if not agent_id:
            return "Error: Agent ID not set in context."

        outcome = outcome.lower().strip()
        if outcome not in ("success", "failure"):
            return f"Invalid outcome '{outcome}'. Use 'success' or 'failure'."

        try:
            from .models import TraitType

            # Find or create skill trait
            trait = await soul_manager.get_trait_by_name(agent_id, skill_name)

            if not trait:
                # Create a new skill trait
                trait = await soul_manager.create_trait(
                    agent_id=agent_id,
                    trait_type=TraitType.SKILL,
                    name=skill_name,
                    description=f"Self-assessed confidence in {skill_name}",
                    expression=f"I have some experience with {skill_name}",
                    strength=0.3,  # Start with low-moderate confidence
                    confidence=0.3,
                )
                logger.info(f"Created new skill trait '{skill_name}' for {agent_id}")

            # Apply outcome
            if outcome == "success":
                updated = await soul_manager.reinforce_trait(
                    agent_id=agent_id,
                    trait_id=trait.id,
                    trigger=f"Skill assessment: {outcome}",
                    evidence=task_context,
                )
                change_word = "increased"
            else:
                updated = await soul_manager.contradict_trait(
                    agent_id=agent_id,
                    trait_id=trait.id,
                    trigger=f"Skill assessment: {outcome}",
                    evidence=task_context,
                )
                change_word = "decreased"

            if updated:
                confidence_level = self._confidence_level(updated.strength)
                return (
                    f"Skill confidence in '{skill_name}' {change_word}.\n"
                    f"Confidence: {trait.strength:.2f} → {updated.strength:.2f} ({confidence_level})"
                )
            else:
                return f"Could not update skill confidence for '{skill_name}'."

        except Exception as e:
            logger.error(f"Error assessing skill: {e}", exc_info=True)
            return f"Error assessing skill: {str(e)}"

    def _confidence_level(self, strength: float) -> str:
        """Convert strength to confidence level description."""
        if strength >= 0.8:
            return "high confidence"
        elif strength >= 0.5:
            return "moderate confidence"
        elif strength >= 0.3:
            return "developing"
        else:
            return "low confidence"


class ViewSoulTool(Tool):
    """
    View your current soul traits and their strengths.

    Provides a snapshot of the agent's personality state.
    """

    def __init__(self, soul_manager: Optional['SoulManager'] = None):
        self._soul_manager = soul_manager

    @property
    def name(self) -> str:
        return "view_soul"

    @property
    def description(self) -> str:
        return (
            "View all your current soul traits and their strengths. "
            "Use this to understand your current personality state, "
            "see what traits are strong or weak, and plan introspection."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """View current soul state."""
        soul_manager = self._soul_manager or getattr(context, 'soul_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        if not soul_manager:
            return "Soul system is not available. Configure REDIS_URL to enable personality persistence."

        if not agent_id:
            return "Error: Agent ID not set in context."

        try:
            soul_context = await soul_manager.build_soul_context(agent_id)

            if not soul_context.has_traits():
                return (
                    "Your soul is still forming. "
                    "You have no established traits yet - they will develop through your interactions."
                )

            lines = [f"**Soul State for {agent_id}:**\n"]

            all_traits = soul_context.get_all_traits()

            # Group by type
            by_type = {}
            for trait in all_traits:
                type_name = trait.trait_type.value
                if type_name not in by_type:
                    by_type[type_name] = []
                by_type[type_name].append(trait)

            type_labels = {
                "core_value": "Core Values",
                "behavioral": "Behavioral Traits",
                "preference": "Learned Preferences",
                "skill": "Skill Confidence",
                "relationship": "Relationships",
            }

            for type_key, type_label in type_labels.items():
                if type_key in by_type:
                    lines.append(f"**{type_label}:**")
                    for trait in sorted(by_type[type_key], key=lambda t: t.strength, reverse=True):
                        stability_emoji = {
                            "novice": "🌱",
                            "developing": "🌿",
                            "established": "🌳",
                            "core": "💎",
                        }.get(trait.stability.value, "")
                        lines.append(
                            f"  {stability_emoji} {trait.name}: {trait.strength:.2f} "
                            f"[{trait.stability.value}]"
                        )
                        lines.append(f"     {trait.expression[:80]}...")
                    lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error viewing soul: {e}", exc_info=True)
            return f"Error viewing soul: {str(e)}"


# Convenience function to get all soul tools
def get_soul_tools(soul_manager: Optional['SoulManager'] = None) -> List[Tool]:
    """Get all soul tools with the given manager."""
    return [
        IntrospectSoulTool(soul_manager),
        EvolveTrait(soul_manager),
        AssessSkillTool(soul_manager),
        ViewSoulTool(soul_manager),
    ]
