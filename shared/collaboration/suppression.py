"""Session-level agent suppression for runtime instructions.

Handles commands like "@Canopus stop testing" or "everyone stand down"
to temporarily suppress agent activity in a channel.
"""

import re
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SuppressionRule:
    """A rule suppressing an agent's activity in a channel."""
    agent_name: str          # Agent name or "*" for all agents
    reason: str              # Why suppression was triggered
    expires_at: datetime     # When suppression ends
    channel_id: int          # Which channel this applies to
    activity_pattern: Optional[str] = None  # Optional: specific activity like "testing"


class SuppressionManager:
    """
    Singleton manager for session-level agent suppression.

    When a user says "Canopus, stop testing" or "everyone stand down",
    this manager tracks the suppression and blocks chime-in for the
    specified duration.
    """
    _instance: Optional['SuppressionManager'] = None

    def __init__(self):
        self._rules: Dict[str, List[SuppressionRule]] = {}  # agent_name -> rules

    @classmethod
    def get_instance(cls) -> 'SuppressionManager':
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def add_suppression(
        self,
        agent_name: str,
        reason: str,
        duration_seconds: int,
        channel_id: int,
        activity_pattern: Optional[str] = None,
    ) -> SuppressionRule:
        """
        Add a suppression rule for an agent.

        Args:
            agent_name: Name of agent to suppress, or "*" for all
            reason: Human-readable reason for the suppression
            duration_seconds: How long to suppress (default: 600 = 10 min)
            channel_id: Which channel this applies to
            activity_pattern: Optional specific activity to suppress

        Returns:
            The created SuppressionRule
        """
        agent_key = agent_name.lower()
        rule = SuppressionRule(
            agent_name=agent_key,
            reason=reason,
            expires_at=datetime.now() + timedelta(seconds=duration_seconds),
            channel_id=channel_id,
            activity_pattern=activity_pattern,
        )

        if agent_key not in self._rules:
            self._rules[agent_key] = []
        self._rules[agent_key].append(rule)

        logger.info(
            f"Suppression added: {agent_name} in channel {channel_id} "
            f"for {duration_seconds}s - {reason}"
        )
        return rule

    def is_suppressed(
        self,
        agent_name: str,
        channel_id: int,
    ) -> Optional[SuppressionRule]:
        """
        Check if an agent is suppressed in a channel.

        Args:
            agent_name: Name of the agent to check
            channel_id: Channel to check suppression in

        Returns:
            The active SuppressionRule if suppressed, None otherwise
        """
        self._cleanup_expired()

        agent_key = agent_name.lower()

        # Check both specific agent and wildcard
        for key in [agent_key, "*"]:
            if key in self._rules:
                for rule in self._rules[key]:
                    if rule.channel_id == channel_id:
                        return rule

        return None

    def clear_suppression(
        self,
        agent_name: str,
        channel_id: int,
    ) -> bool:
        """
        Clear suppression for an agent in a channel.

        Args:
            agent_name: Name of agent to clear, or "*" for all
            channel_id: Channel to clear suppression in

        Returns:
            True if any rules were cleared
        """
        agent_key = agent_name.lower()
        cleared = False

        if agent_key in self._rules:
            original_count = len(self._rules[agent_key])
            self._rules[agent_key] = [
                r for r in self._rules[agent_key]
                if r.channel_id != channel_id
            ]
            cleared = len(self._rules[agent_key]) < original_count

            if not self._rules[agent_key]:
                del self._rules[agent_key]

        if cleared:
            logger.info(f"Suppression cleared: {agent_name} in channel {channel_id}")

        return cleared

    def list_suppressions(self, channel_id: Optional[int] = None) -> List[SuppressionRule]:
        """List all active suppressions, optionally filtered by channel."""
        self._cleanup_expired()

        rules = []
        for rule_list in self._rules.values():
            for rule in rule_list:
                if channel_id is None or rule.channel_id == channel_id:
                    rules.append(rule)

        return rules

    def _cleanup_expired(self) -> None:
        """Remove expired suppression rules."""
        now = datetime.now()
        for agent_name in list(self._rules.keys()):
            self._rules[agent_name] = [
                r for r in self._rules[agent_name] if r.expires_at > now
            ]
            if not self._rules[agent_name]:
                del self._rules[agent_name]


# Common suppression command patterns
# NOTE: Requires @ mention to avoid false positives from casual conversation
SUPPRESSION_PATTERNS = [
    # Direct commands with Discord mention: "<@123> stop", "<@123> stand down"
    (r"<@!?(\d+)>,?\s*(stop|stand\s*down|don'?t\s+respond|be\s+quiet|shut\s+up)", "direct_mention"),
    # Direct commands with @ prefix: "@Canopus stop", "@vega stand down"
    (r"@(\w+),?\s+(stop|stand\s*down|don'?t\s+respond|be\s+quiet|shut\s+up)", "direct_at"),
    # Activity-specific with mention: "<@123> stop testing"
    (r"<@!?(\d+)>,?\s*stop\s+(\w+ing)", "activity_mention"),
    # Activity-specific with @: "@Canopus stop testing"
    (r"@(\w+),?\s+stop\s+(\w+ing)", "activity_at"),
    # Group commands: "everyone stop", "all agents stand down"
    (r"(everyone|all\s+agents?|all\s+of\s+you)\s+(stop|stand\s*down|be\s+quiet)", "group"),
    # Resume with Discord mention: "<@123> resume"
    (r"<@!?(\d+)>,?\s*(resume|you\s+can\s+respond|speak\s+again|continue)", "resume_mention"),
    # Resume with @: "@Canopus resume"
    (r"@(\w+),?\s+(resume|you\s+can\s+respond|speak\s+again|continue)", "resume_at"),
]


def _id_to_agent_name(user_id: str, agent_registry: Dict[str, int]) -> Optional[str]:
    """Convert a Discord user ID to agent name using the registry."""
    try:
        uid = int(user_id)
        for name, registered_id in agent_registry.items():
            if registered_id == uid:
                return name.lower()
    except ValueError:
        pass
    return None


def parse_suppression_command(
    content: str,
    agent_registry: Dict[str, int],
) -> Optional[Dict]:
    """
    Parse a message for suppression commands.

    Requires @ mention (either Discord <@123> or @AgentName) to trigger.
    This prevents false positives from casual conversation.

    Args:
        content: Message content to parse
        agent_registry: Dict mapping agent names to Discord user IDs

    Returns:
        Dict with parsed command info, or None if not a suppression command
        {
            "action": "suppress" | "resume",
            "agent": "agent_name" | "*",
            "activity": Optional activity pattern (e.g., "testing"),
            "duration": seconds (default 600),
        }
    """
    for pattern, pattern_type in SUPPRESSION_PATTERNS:
        match = re.search(pattern, content, re.IGNORECASE)
        if not match:
            continue

        # Handle Discord mention patterns (capture user ID)
        if pattern_type == "direct_mention":
            user_id = match.group(1)
            agent = _id_to_agent_name(user_id, agent_registry)
            if not agent:
                continue
            return {
                "action": "suppress",
                "agent": agent,
                "activity": None,
                "duration": 600,
            }

        elif pattern_type == "direct_at":
            agent = match.group(1).lower()
            if agent not in agent_registry:
                continue
            return {
                "action": "suppress",
                "agent": agent,
                "activity": None,
                "duration": 600,
            }

        elif pattern_type == "activity_mention":
            user_id = match.group(1)
            activity = match.group(2).lower()
            agent = _id_to_agent_name(user_id, agent_registry)
            if not agent:
                continue
            return {
                "action": "suppress",
                "agent": agent,
                "activity": activity,
                "duration": 600,
            }

        elif pattern_type == "activity_at":
            agent = match.group(1).lower()
            activity = match.group(2).lower()
            if agent not in agent_registry:
                continue
            return {
                "action": "suppress",
                "agent": agent,
                "activity": activity,
                "duration": 600,
            }

        elif pattern_type == "group":
            return {
                "action": "suppress",
                "agent": "*",
                "activity": None,
                "duration": 600,
            }

        elif pattern_type == "resume_mention":
            user_id = match.group(1)
            agent = _id_to_agent_name(user_id, agent_registry)
            if not agent:
                continue
            return {
                "action": "resume",
                "agent": agent,
                "activity": None,
                "duration": 0,
            }

        elif pattern_type == "resume_at":
            agent = match.group(1).lower()
            if agent not in agent_registry:
                continue

            return {
                "action": "resume",
                "agent": agent,
                "activity": None,
                "duration": 0,
            }

    return None
