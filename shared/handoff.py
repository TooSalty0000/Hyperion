"""Scalable agent handoff evaluation using LLM-based agent discovery."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from shared.llm.types import Message, Role

if TYPE_CHECKING:
    from shared.llm.interface import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class HandoffSuggestion:
    """A suggested agent handoff."""
    agent_name: str
    reason: str
    confidence: float  # 0.0 to 1.0


class HandoffEvaluator:
    """
    Evaluate which agents should receive findings - scales with new agents.

    This uses LLM to intelligently match findings to agents based on their
    description (chime_in_description). When adding a new agent, simply
    register its description and the system will automatically consider
    it for handoffs.

    Features:
    - Scalable: Just add agents with good descriptions
    - LLM-based: Uses natural language understanding
    - Context-aware: Considers what the findings actually contain
    - Efficient: Single LLM call to evaluate all registered agents

    Usage:
        evaluator = HandoffEvaluator(llm_provider)
        evaluator.register_agent("polaris", "Calendar and scheduling specialist...")
        evaluator.register_agent("altair", "Project management and code specialist...")

        suggestions = await evaluator.suggest_handoff(
            findings="Found show dates: Dec 2025 - Mar 2026",
            own_agent="canopus"
        )
        # Returns: [HandoffSuggestion(agent_name="polaris", reason="...", confidence=0.9)]
    """

    def __init__(self, llm: 'LLMProvider'):
        """
        Initialize the HandoffEvaluator.

        Args:
            llm: LLM provider for evaluating handoffs
        """
        self.llm = llm
        self._descriptions: Dict[str, str] = {}

    def register_agent(self, name: str, description: str) -> None:
        """
        Register an agent's expertise description for handoff evaluation.

        The description should explain what kinds of tasks/information
        the agent specializes in.

        Args:
            name: Agent name (lowercase)
            description: What the agent specializes in
        """
        self._descriptions[name.lower()] = description
        logger.debug(f"Registered agent '{name}' for handoff evaluation")

    def unregister_agent(self, name: str) -> None:
        """Remove an agent from handoff evaluation."""
        self._descriptions.pop(name.lower(), None)

    @property
    def registered_agents(self) -> List[str]:
        """List of registered agent names."""
        return list(self._descriptions.keys())

    def get_description(self, agent_name: str) -> Optional[str]:
        """Get an agent's registered description."""
        return self._descriptions.get(agent_name.lower())

    async def suggest_handoff(
        self,
        findings: str,
        own_agent: str,
        context: Optional[str] = None,
        max_suggestions: int = 2,
    ) -> List[HandoffSuggestion]:
        """
        Use LLM to suggest which agents should receive these findings.

        Args:
            findings: The information/findings to potentially hand off
            own_agent: Name of the agent doing the handoff (excluded from suggestions)
            context: Optional additional context about the task
            max_suggestions: Maximum number of agents to suggest

        Returns:
            List of HandoffSuggestion with agent names, reasons, and confidence
        """
        # Filter out the calling agent
        other_agents = {
            name: desc
            for name, desc in self._descriptions.items()
            if name.lower() != own_agent.lower()
        }

        if not other_agents:
            logger.debug("No other agents registered for handoff evaluation")
            return []

        # Build the evaluation prompt
        agents_description = "\n".join(
            f"- **{name}**: {desc}"
            for name, desc in other_agents.items()
        )

        prompt = f"""Analyze these findings and determine which agents (if any) should be notified.

## Findings from {own_agent}:
{findings}

{f"## Additional Context: {context}" if context else ""}

## Available Agents:
{agents_description}

## Instructions:
1. Evaluate if ANY agent's expertise DIRECTLY relates to these findings
2. Only suggest agents whose domain is clearly relevant
3. If no agent is clearly relevant, respond with "NONE"

## Response Format:
For each suggested agent, respond with one line:
AGENT_NAME: REASON (CONFIDENCE: high/medium/low)

Or if no handoff needed:
NONE: No agent's expertise directly matches these findings

Examples:
- "polaris: Contains event dates that should be added to calendar (CONFIDENCE: high)"
- "altair: Involves project task that needs tracking (CONFIDENCE: medium)"
- "NONE: Findings are informational only, no action needed"
"""

        try:
            response = await self.llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                max_tokens=300,
                temperature=0.3,  # Lower temperature for more consistent evaluation
            )

            if not response.content:
                logger.warning("Empty response from LLM for handoff evaluation")
                return []

            return self._parse_suggestions(response.content, max_suggestions)

        except Exception as e:
            logger.error(f"Handoff evaluation failed: {e}")
            return []

    def _parse_suggestions(
        self,
        response: str,
        max_suggestions: int
    ) -> List[HandoffSuggestion]:
        """Parse LLM response into HandoffSuggestion objects."""
        suggestions = []

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for "NONE" response
            if line.upper().startswith("NONE"):
                return []

            # Parse "AGENT_NAME: REASON (CONFIDENCE: level)"
            if ":" not in line:
                continue

            parts = line.split(":", 1)
            agent_name = parts[0].strip().lower()

            # Skip if not a registered agent
            if agent_name not in self._descriptions:
                # Could be a non-agent line like "Example:"
                continue

            reason_part = parts[1].strip() if len(parts) > 1 else ""

            # Extract confidence
            confidence = 0.5  # Default medium
            confidence_str = reason_part.lower()

            if "confidence: high" in confidence_str or "(high)" in confidence_str:
                confidence = 0.9
            elif "confidence: medium" in confidence_str or "(medium)" in confidence_str:
                confidence = 0.6
            elif "confidence: low" in confidence_str or "(low)" in confidence_str:
                confidence = 0.3

            # Clean up reason (remove confidence notation)
            reason = reason_part
            for marker in ["(CONFIDENCE: high)", "(CONFIDENCE: medium)", "(CONFIDENCE: low)",
                          "(high)", "(medium)", "(low)", "CONFIDENCE: high",
                          "CONFIDENCE: medium", "CONFIDENCE: low"]:
                reason = reason.replace(marker, "").replace(marker.lower(), "")
            reason = reason.strip().rstrip("(").strip()

            suggestions.append(HandoffSuggestion(
                agent_name=agent_name,
                reason=reason if reason else "Relevant expertise",
                confidence=confidence,
            ))

            if len(suggestions) >= max_suggestions:
                break

        # Sort by confidence (highest first)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return suggestions

    async def should_handoff(
        self,
        findings: str,
        own_agent: str,
        confidence_threshold: float = 0.5,
    ) -> bool:
        """
        Quick check if a handoff is likely needed.

        Args:
            findings: The findings to evaluate
            own_agent: The agent checking
            confidence_threshold: Minimum confidence to return True

        Returns:
            True if at least one agent should receive the findings
        """
        suggestions = await self.suggest_handoff(findings, own_agent, max_suggestions=1)
        return bool(suggestions and suggestions[0].confidence >= confidence_threshold)


def create_handoff_message(
    findings: str,
    suggestion: HandoffSuggestion,
    from_agent: str,
    task_summary: Optional[str] = None,
) -> str:
    """
    Create a formatted handoff message for Discord.

    Args:
        findings: The findings to hand off
        suggestion: The HandoffSuggestion with target agent
        from_agent: Name of the sending agent
        task_summary: Optional summary of what was accomplished

    Returns:
        Formatted message string (without the @mention - that's added by the tool)
    """
    parts = []

    if task_summary:
        parts.append(f"**Task Completed:** {task_summary}")

    parts.append(f"**Findings for you:**\n{findings}")
    parts.append(f"\n*Reason for handoff: {suggestion.reason}*")

    return "\n".join(parts)
