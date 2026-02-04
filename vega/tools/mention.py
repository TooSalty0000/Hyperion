"""Mention tools for inter-agent communication via Discord @mentions."""

import asyncio
import logging
from typing import List, TYPE_CHECKING, Optional

import discord

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

if TYPE_CHECKING:
    from discord.ext import commands
    from shared.agent_coordinator import DistributedAgentTracker

logger = logging.getLogger(__name__)


class MentionAgentTool(Tool):
    """
    Mention another agent in Discord to communicate with them.

    This posts a visible @mention message in the current channel.
    The target agent will see the mention and can respond.

    Features:
    - Status inference: Checks if target agent has been active recently
    - Delivery tracking: Tracks sent messages and watches for acknowledgment reactions
    - Retry logic: Automatically retries failed deliveries
    - Optional waiting: Can wait for the agent to acknowledge (via ✅ reaction)

    Communication Protocol:
    - When this tool sends a message, it adds a `[from: agent_name]` marker
    - Target agent adds a ✅ reaction when they receive and start processing
    - Target agent adds a ✔️ reaction when processing is complete
    - All agents can observe these reactions to infer who's online
    """

    def __init__(
        self,
        bot: 'commands.Bot',
        agent_registry: dict[str, int],
        own_agent_name: str = "vega",
        tracker: 'DistributedAgentTracker' = None,
    ):
        """
        Initialize MentionAgentTool.

        Args:
            bot: Discord bot instance for sending messages
            agent_registry: Mapping of agent names to Discord user IDs
            own_agent_name: Name of the agent using this tool
            tracker: Optional DistributedAgentTracker for tracking/acknowledgment
        """
        self._bot = bot
        self._agent_registry = agent_registry
        self._own_name = own_agent_name
        self._tracker = tracker

    def set_tracker(self, tracker: 'DistributedAgentTracker') -> None:
        """Set the distributed tracker after initialization."""
        self._tracker = tracker

    @property
    def name(self) -> str:
        return "mention_agent"

    @property
    def description(self) -> str:
        available_agents = [
            name for name in self._agent_registry.keys()
            if name != self._own_name
        ]
        agents_str = ", ".join(available_agents) if available_agents else "none configured"

        return (
            f"Send a visible @mention to another agent in the Discord channel. "
            f"The message will be visible to the user and the target agent will respond. "
            f"Use this to delegate work, ask questions, or request status from other agents. "
            f"Available agents: {agents_str}. "
            f"The system will check if the agent has been active recently."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="agent_name",
                type="string",
                description="Name of the agent to mention (e.g., 'altair', 'polaris', 'canopus')",
                required=True,
            ),
            ToolParameter(
                name="message",
                type="string",
                description="The message to send to the agent",
                required=True,
            ),
            ToolParameter(
                name="request_type",
                type="string",
                description="Type of request to help the agent understand context",
                required=False,
                enum=["action", "status", "question", "task"],
            ),
            ToolParameter(
                name="project_name",
                type="string",
                description="Optional project name for context",
                required=False,
            ),
            ToolParameter(
                name="wait_for_ack",
                type="boolean",
                description="If true, wait for the agent to acknowledge receipt (max 300s). Default: false",
                required=False,
            ),
            ToolParameter(
                name="urgent",
                type="boolean",
                description="Mark as urgent - agent will prioritize this message",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        agent_name: str,
        message: str,
        request_type: str = None,
        project_name: str = None,
        wait_for_ack: bool = False,
        urgent: bool = False,
    ) -> str:
        """Send a @mention to another agent with delivery tracking."""
        agent_name = agent_name.lower()

        # Validate agent exists
        if agent_name not in self._agent_registry:
            available = ", ".join(
                n for n in self._agent_registry.keys()
                if n != self._own_name
            )
            return f"Unknown agent '{agent_name}'. Available agents: {available}"

        # Can't mention yourself
        if agent_name == self._own_name:
            return "You cannot mention yourself."

        # Get channel
        if not context.current_channel_id:
            return "No channel context available to send message."

        channel = self._bot.get_channel(context.current_channel_id)
        if not channel:
            return f"Could not find channel {context.current_channel_id}."

        # Check if agent has been active recently (if tracker available)
        status_warning = ""
        if self._tracker:
            status = self._tracker.get_agent_status(agent_name)
            if status:
                if status["status"] == "possibly_offline":
                    status_warning = (
                        f" (Note: {agent_name} hasn't been active for "
                        f"{status['seconds_since_activity']}s - they may not respond immediately)"
                    )
                elif status["status"] == "unknown":
                    status_warning = f" (Note: {agent_name}'s status is unknown)"

        # Build the mention message
        agent_user_id = self._agent_registry[agent_name]
        mention = f"<@{agent_user_id}>"

        # Format message with optional context
        formatted_message = message
        if urgent:
            formatted_message = f"[URGENT] {formatted_message}"
        if request_type:
            formatted_message = f"[{request_type.upper()}] {formatted_message}"
        if project_name:
            formatted_message = f"[Project: {project_name}] {formatted_message}"

        # Add source agent marker for tracking (helps target agent identify source)
        formatted_message = f"{formatted_message}\n`[from: {self._own_name}]`"

        full_message = f"{mention} {formatted_message}"

        # Send the message with retry logic
        sent_msg = await self._send_with_retry(channel, full_message, max_retries=2)

        if not sent_msg:
            return (
                f"Failed to send message to @{agent_name} after multiple attempts. "
                f"Please try again later."
            )

        # Track the outgoing message (if tracker available)
        if self._tracker:
            self._tracker.track_outgoing(sent_msg.id, agent_name, message)

            # If waiting for acknowledgment, block until ✅ reaction or timeout
            if wait_for_ack:
                logger.info(
                    f"[MentionAgentTool] Waiting for acknowledgment from {agent_name}..."
                )

                acknowledged = await self._tracker.wait_for_acknowledgment(
                    sent_msg.id,
                    timeout=300.0
                )

                if acknowledged:
                    return (
                        f"✓ @{agent_name} has acknowledged the message and is processing it. "
                        f"(Do not repeat this message in your response.)"
                    )
                else:
                    return (
                        f"✓ Message sent to @{agent_name}, but no acknowledgment received yet. "
                        f"They may still respond shortly.{status_warning} "
                        f"(Do not repeat this message in your response.)"
                    )

        logger.info(
            f"[MentionAgentTool] MENTION SENT: target={agent_name}, "
            f"sent_msg_id={sent_msg.id}, channel={channel.id}, "
            f"content={full_message[:80]}..."
        )

        # Return a clear, non-duplicative response
        return (
            f"✓ @{agent_name} has been notified. They will respond shortly.{status_warning} "
            f"(Do not repeat this message in your response.)"
        )

    async def _send_with_retry(
        self,
        channel: discord.TextChannel,
        content: str,
        max_retries: int = 2
    ) -> Optional[discord.Message]:
        """Send a message with retry logic."""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                sent_msg = await channel.send(content, silent=True)
                return sent_msg

            except discord.Forbidden as e:
                logger.error(
                    f"[MentionAgentTool] PERMISSION DENIED: channel={channel.id}, "
                    f"attempt={attempt + 1}"
                )
                last_error = e
                break  # Don't retry permission errors

            except discord.HTTPException as e:
                logger.warning(
                    f"[MentionAgentTool] HTTP ERROR: {e}, attempt={attempt + 1}/{max_retries + 1}"
                )
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

            except Exception as e:
                logger.error(f"[MentionAgentTool] UNEXPECTED ERROR: {e}")
                last_error = e
                break

        logger.error(
            f"[MentionAgentTool] Failed to send after {max_retries + 1} attempts: {last_error}"
        )
        return None


class CheckAgentStatusTool(Tool):
    """
    Check the status of other agents in the system.

    Status is inferred from observed Discord activity:
    - active: Agent has been active in the last 2 minutes
    - likely_online: Agent was active in the last 10 minutes
    - possibly_offline: No activity for 10+ minutes
    - unknown: Never observed any activity
    """

    def __init__(
        self,
        agent_registry: dict[str, int],
        own_agent_name: str = "vega",
        tracker: 'DistributedAgentTracker' = None,
    ):
        self._agent_registry = agent_registry
        self._own_name = own_agent_name
        self._tracker = tracker

    def set_tracker(self, tracker: 'DistributedAgentTracker') -> None:
        """Set the distributed tracker after initialization."""
        self._tracker = tracker

    @property
    def name(self) -> str:
        return "check_agent_status"

    @property
    def description(self) -> str:
        return (
            "Check the status of other agents based on their recent Discord activity. "
            "Returns whether agents appear to be online, their last activity time, "
            "and any pending messages awaiting acknowledgment."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="agent_name",
                type="string",
                description="Specific agent to check (optional - leave empty for all agents)",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        agent_name: str = None,
    ) -> str:
        """Check agent status based on observed activity."""
        if not self._tracker:
            # Fallback: list registered agents without status info
            agents = [n for n in self._agent_registry.keys() if n != self._own_name]
            return (
                f"Agent status tracking not available. "
                f"Registered agents: {', '.join(agents)}"
            )

        if agent_name:
            # Check specific agent
            agent_name = agent_name.lower()
            status = self._tracker.get_agent_status(agent_name)

            if not status:
                if agent_name in self._agent_registry:
                    return f"Agent '{agent_name}' is registered but no activity observed yet."
                return f"Unknown agent: {agent_name}"

            lines = [f"**{agent_name.title()} Status**"]

            status_emoji = {
                "active": "🟢",
                "likely_online": "🟡",
                "possibly_offline": "🔴",
                "unknown": "⚪",
            }.get(status["status"], "⚪")

            lines.append(f"- Status: {status_emoji} {status['status']}")

            if status["seconds_since_activity"] is not None:
                mins = status["seconds_since_activity"] // 60
                secs = status["seconds_since_activity"] % 60
                if mins > 0:
                    lines.append(f"- Last activity: {mins}m {secs}s ago")
                else:
                    lines.append(f"- Last activity: {secs}s ago")
            else:
                lines.append("- Last activity: never observed")

            if status["pending_messages"] > 0:
                lines.append(f"- Pending messages: {status['pending_messages']}")

            return "\n".join(lines)

        # Check all agents
        all_status = self._tracker.get_all_status()

        if not all_status:
            agents = [n for n in self._agent_registry.keys() if n != self._own_name]
            return f"No agent activity observed yet. Registered: {', '.join(agents)}"

        lines = ["**Agent Status Overview**"]

        for name, status in sorted(all_status.items()):
            if status is None:
                continue

            status_emoji = {
                "active": "🟢",
                "likely_online": "🟡",
                "possibly_offline": "🔴",
                "unknown": "⚪",
            }.get(status["status"], "⚪")

            line = f"- {status_emoji} **{name.title()}**: {status['status']}"

            if status["seconds_since_activity"] is not None:
                mins = status["seconds_since_activity"] // 60
                if mins > 0:
                    line += f" ({mins}m ago)"
                else:
                    line += f" ({status['seconds_since_activity']}s ago)"

            if status["pending_messages"] > 0:
                line += f" [📨 {status['pending_messages']} pending]"

            lines.append(line)

        # Add any registered but never-observed agents
        for name in self._agent_registry.keys():
            if name != self._own_name and name not in all_status:
                lines.append(f"- ⚪ **{name.title()}**: never observed")

        return "\n".join(lines)
