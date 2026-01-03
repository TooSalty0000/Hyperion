"""Cognitive tools for Canopus navigation planning and progress tracking."""

import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

if TYPE_CHECKING:
    from discord.ext import commands
    from shared.llm.interface import LLMProvider

logger = logging.getLogger(__name__)


class PlanNavigationTool(Tool):
    """Create a step-by-step plan for achieving the navigation goal."""

    @property
    def name(self) -> str:
        return "plan_navigation"

    @property
    def description(self) -> str:
        return (
            "Create a step-by-step plan for achieving the navigation goal. "
            "ALWAYS use this tool FIRST before any browser actions. "
            "Break down the goal into clear, actionable steps. "
            "Each step should be a single action (navigate, click, type, extract)."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="goal",
                type="string",
                description="The navigation goal to achieve (e.g., 'Find pricing for GitHub Copilot')",
                required=True,
            ),
            ToolParameter(
                name="steps",
                type="array",
                description="List of step descriptions to achieve the goal",
                required=True,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        goal: str,
        steps: List[str],
    ) -> str:
        """Create navigation plan."""
        if not hasattr(context, 'scratchpad') or context.scratchpad is None:
            return "Error: Scratchpad not available."

        if not goal:
            return "Error: Goal cannot be empty."

        if not steps or len(steps) == 0:
            return "Error: Must provide at least one step."

        # Store in scratchpad
        context.scratchpad.set_goal(goal)
        context.scratchpad.set_plan(steps)

        # Format response
        lines = [
            f"✅ Navigation plan created!",
            f"",
            f"Goal: {goal}",
            f"",
            f"Steps ({len(steps)}):"
        ]
        for i, step in enumerate(steps, 1):
            marker = "→" if i == 1 else "○"
            lines.append(f"  {i}. {marker} {step}")

        lines.append("")
        lines.append("Now use get_snapshot to observe the current page.")

        return "\n".join(lines)


class StoreFindingTool(Tool):
    """Store discovered information in the scratchpad."""

    @property
    def name(self) -> str:
        return "store_finding"

    @property
    def description(self) -> str:
        return (
            "Store important information discovered during navigation. "
            "Use this to remember prices, names, URLs, or any data needed later. "
            "Each finding has a key for easy retrieval."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="key",
                type="string",
                description="A short identifier for this finding (e.g., 'pricing', 'product_name')",
                required=True,
            ),
            ToolParameter(
                name="value",
                type="string",
                description="The information to store",
                required=True,
            ),
            ToolParameter(
                name="finding_context",
                type="string",
                description="Optional context about where/how this was found",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        key: str,
        value: str,
        finding_context: Optional[str] = None,
    ) -> str:
        """Store a finding."""
        if not hasattr(context, 'scratchpad') or context.scratchpad is None:
            return "Error: Scratchpad not available."

        if not key or not value:
            return "Error: Both key and value are required."

        # Get current URL if browser session available
        current_url = ""
        if hasattr(context, 'browser_manager') and context.browser_manager:
            try:
                session = await context.browser_manager.get(context.current_channel_id)
                if session:
                    current_url = session.current_url or ""
            except Exception:
                pass

        context.scratchpad.add_finding(
            key=key,
            value=value,
            url=current_url,
            context=finding_context
        )

        count = len(context.scratchpad.findings)
        return f"✅ Stored finding '{key}': {value}\n(Total findings: {count})"


class ReadFindingsTool(Tool):
    """Read all stored findings from the scratchpad."""

    @property
    def name(self) -> str:
        return "read_findings"

    @property
    def description(self) -> str:
        return (
            "Read all information stored during navigation. "
            "Use this to recall what you've discovered. "
            "Optionally filter by key to find specific information."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="key",
                type="string",
                description="Optional: specific finding key to retrieve",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        key: Optional[str] = None,
    ) -> str:
        """Read findings."""
        if not hasattr(context, 'scratchpad') or context.scratchpad is None:
            return "Error: Scratchpad not available."

        if key:
            finding = context.scratchpad.get_finding(key)
            if finding:
                lines = [
                    f"Finding '{key}':",
                    f"  Value: {finding.value}",
                ]
                if finding.url:
                    lines.append(f"  URL: {finding.url}")
                if finding.context:
                    lines.append(f"  Context: {finding.context}")
                return "\n".join(lines)
            else:
                return f"No finding with key '{key}' found."

        return context.scratchpad.get_findings_formatted()


class CheckProgressTool(Tool):
    """Check current progress against the navigation plan."""

    @property
    def name(self) -> str:
        return "check_progress"

    @property
    def description(self) -> str:
        return (
            "Check your current progress in the navigation plan. "
            "Shows completed steps, current step, and remaining steps. "
            "Use this to verify where you are and what to do next."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="mark_current_complete",
                type="boolean",
                description="If true, mark the current step as complete and advance",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        mark_current_complete: bool = False,
    ) -> str:
        """Check progress."""
        if not hasattr(context, 'scratchpad') or context.scratchpad is None:
            return "Error: Scratchpad not available."

        scratchpad = context.scratchpad

        if mark_current_complete:
            current = scratchpad.get_current_step()
            if current:
                next_step = scratchpad.mark_step_complete()
                if next_step:
                    return f"✓ Completed: {current}\n→ Next: {next_step}"
                else:
                    return f"✓ Completed: {current}\n\n🎉 All steps complete! Use mark_goal_complete to finish."

        return scratchpad.get_progress_summary()


class MarkGoalCompleteTool(Tool):
    """Mark the navigation goal as achieved and optionally hand off findings."""

    def __init__(
        self,
        discord_bot: Optional['commands.Bot'] = None,
        agent_registry: Optional[Dict[str, int]] = None,
        llm: Optional['LLMProvider'] = None,
        agent_descriptions: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize MarkGoalCompleteTool.

        Args:
            discord_bot: Discord bot for sending handoff messages
            agent_registry: Mapping of agent names to Discord user IDs
            llm: LLM provider for auto-evaluating handoffs
            agent_descriptions: Agent name to description mapping for auto-discovery
        """
        self._bot = discord_bot
        self._agent_registry = agent_registry or {}
        self._llm = llm
        self._agent_descriptions = agent_descriptions or {}

    def register_agent_description(self, name: str, description: str) -> None:
        """Register an agent's description for handoff evaluation."""
        self._agent_descriptions[name.lower()] = description

    @property
    def name(self) -> str:
        return "mark_goal_complete"

    @property
    def description(self) -> str:
        return (
            "Mark the navigation goal as achieved (or failed). "
            "Use this when you have accomplished the goal or determined it's not achievable. "
            "Provide a summary of what was accomplished or why it failed. "
            "Optionally, hand off findings to another agent who might need them. "
            "Use handoff_to='auto' to let the system suggest who should receive the findings."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        # Build list of available agents for handoff
        available_agents = [
            name for name in self._agent_registry.keys()
            if name.lower() != "canopus"
        ]
        handoff_options = ", ".join(available_agents) if available_agents else "none configured"

        return [
            ToolParameter(
                name="success",
                type="boolean",
                description="True if goal was achieved, False if failed",
                required=True,
            ),
            ToolParameter(
                name="summary",
                type="string",
                description="Summary of what was accomplished or why it failed",
                required=True,
            ),
            ToolParameter(
                name="handoff_to",
                type="string",
                description=(
                    f"Optional: Agent(s) to hand off findings to. "
                    f"Use 'auto' to let the system suggest based on findings content. "
                    f"Use specific name(s) like 'polaris' or 'altair,polaris' for explicit handoff. "
                    f"Available agents: {handoff_options}"
                ),
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        success: bool,
        summary: str,
        handoff_to: Optional[str] = None,
    ) -> str:
        """Mark goal complete and optionally hand off findings."""
        if not hasattr(context, 'scratchpad') or context.scratchpad is None:
            return "Error: Scratchpad not available."

        context.scratchpad.set_goal_achieved(success, summary)

        # Build the base response
        if success:
            lines = [
                "✅ Goal achieved!",
                "",
                f"Summary: {summary}",
                "",
                "Findings collected:",
            ]
            for f in context.scratchpad.findings:
                lines.append(f"  • {f.key}: {f.value}")

            if not context.scratchpad.findings:
                lines.append("  (none)")

            base_response = "\n".join(lines)
        else:
            base_response = f"❌ Goal not achieved.\n\nReason: {summary}"

        # Handle handoff if requested and goal succeeded
        if handoff_to and success:
            handoff_result = await self._handle_handoff(
                context=context,
                findings_summary=summary,
                handoff_to=handoff_to,
            )
            if handoff_result:
                base_response += f"\n\n{handoff_result}"

        return base_response

    async def _handle_handoff(
        self,
        context: ToolContext,
        findings_summary: str,
        handoff_to: str,
    ) -> Optional[str]:
        """Handle handoff to other agents."""
        if not self._bot or not self._agent_registry:
            return "⚠️ Handoff not available (bot/registry not configured)"

        # Build findings string
        findings_parts = [findings_summary]
        if hasattr(context, 'scratchpad') and context.scratchpad:
            for f in context.scratchpad.findings:
                findings_parts.append(f"- {f.key}: {f.value}")
        findings_str = "\n".join(findings_parts)

        # Determine target agents
        target_agents = []

        if handoff_to.lower() == "auto":
            # Use HandoffEvaluator for auto-detection
            target_agents = await self._auto_detect_handoff(findings_str)
        else:
            # Parse explicit agent names
            target_agents = [
                name.strip().lower()
                for name in handoff_to.split(",")
                if name.strip().lower() in self._agent_registry
                and name.strip().lower() != "canopus"
            ]

        if not target_agents:
            return None

        # Send handoff messages
        handoff_results = []
        for agent_name in target_agents:
            result = await self._send_handoff(
                context=context,
                agent_name=agent_name,
                findings=findings_str,
            )
            handoff_results.append(result)

        return "\n".join(handoff_results)

    async def _auto_detect_handoff(self, findings: str) -> List[str]:
        """Use LLM to determine which agents should receive findings."""
        if not self._llm or not self._agent_descriptions:
            logger.debug("Auto-handoff not available (LLM or descriptions not configured)")
            return []

        try:
            from shared.handoff import HandoffEvaluator

            evaluator = HandoffEvaluator(self._llm)
            for name, desc in self._agent_descriptions.items():
                if name.lower() != "canopus":
                    evaluator.register_agent(name, desc)

            suggestions = await evaluator.suggest_handoff(
                findings=findings,
                own_agent="canopus",
                max_suggestions=2,
            )

            # Only return high-confidence suggestions
            return [
                s.agent_name
                for s in suggestions
                if s.confidence >= 0.5
            ]

        except Exception as e:
            logger.error(f"Auto-handoff evaluation failed: {e}")
            return []

    async def _send_handoff(
        self,
        context: ToolContext,
        agent_name: str,
        findings: str,
    ) -> str:
        """Send handoff message to an agent."""
        if not self._bot or not context.current_channel_id:
            return f"⚠️ Could not send handoff to {agent_name}"

        try:
            import discord

            channel = self._bot.get_channel(context.current_channel_id)
            if not channel:
                return f"⚠️ Channel not found for handoff to {agent_name}"

            agent_user_id = self._agent_registry.get(agent_name.lower())
            if not agent_user_id:
                return f"⚠️ Agent {agent_name} not found in registry"

            mention = f"<@{agent_user_id}>"
            handoff_message = (
                f"{mention} [HANDOFF from Canopus]\n\n"
                f"**Findings for you:**\n{findings}\n\n"
                f"`[from: canopus]`"
            )

            await channel.send(handoff_message)
            logger.info(f"[Canopus] Sent handoff to {agent_name}")
            return f"📤 Handed off to @{agent_name}"

        except discord.HTTPException as e:
            logger.error(f"Failed to send handoff to {agent_name}: {e}")
            return f"⚠️ Failed to send handoff to {agent_name}: {e}"
        except Exception as e:
            logger.error(f"Handoff error: {e}")
            return f"⚠️ Handoff error: {e}"
