"""
Distributed Agent Tracker - Decentralized inter-agent communication via Discord.

This module provides reliable agent-to-agent communication in a distributed system
where each agent runs independently (potentially on different servers).

Key Design Principles:
- Discord is the ONLY communication medium
- No shared memory or central coordinator
- Each agent maintains its own local view of other agents
- Acknowledgment uses Discord reactions (visible to all)
- Health is inferred from observed Discord activity

How it works:
1. When Agent A mentions Agent B, it tracks the message locally
2. Agent B sees the mention, processes it, and adds a ✅ reaction
3. Agent A sees the reaction and knows B received the message
4. All agents can observe reactions to infer who's online
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Set, List, TYPE_CHECKING
import re

if TYPE_CHECKING:
    import discord
    from discord.ext import commands

logger = logging.getLogger(__name__)


# Reaction emojis for agent communication protocol
REACTION_ACK = "✅"          # Message received and being processed
REACTION_DONE = "✔️"         # Message fully processed
REACTION_BUSY = "⏳"         # Agent is busy, will process later
REACTION_ERROR = "❌"        # Error processing message
REACTION_SEEN = "👀"         # Message seen but no action needed


class InferredStatus(Enum):
    """Status inferred from observed Discord activity."""
    ACTIVE = "active"           # Recently sent messages or reactions
    LIKELY_ONLINE = "likely_online"   # Has recent activity
    POSSIBLY_OFFLINE = "possibly_offline"  # No recent activity
    UNKNOWN = "unknown"         # Never seen any activity


@dataclass
class AgentObservation:
    """
    Local observation of another agent's activity.

    Each agent maintains its own observations - this is NOT shared state.
    """
    agent_name: str
    user_id: int  # Discord user ID from registry

    # Last observed activity (all local timestamps)
    last_message_seen: Optional[datetime] = None
    last_reaction_seen: Optional[datetime] = None
    last_mention_response: Optional[datetime] = None

    # Tracking for messages we sent TO this agent
    pending_messages: Dict[int, datetime] = field(default_factory=dict)  # msg_id -> sent_at
    acknowledged_messages: Set[int] = field(default_factory=set)

    @property
    def last_activity(self) -> Optional[datetime]:
        """Most recent observed activity."""
        times = [t for t in [
            self.last_message_seen,
            self.last_reaction_seen,
            self.last_mention_response
        ] if t is not None]
        return max(times) if times else None

    @property
    def inferred_status(self) -> InferredStatus:
        """Infer agent status from observed activity."""
        if not self.last_activity:
            return InferredStatus.UNKNOWN

        age = datetime.now() - self.last_activity

        if age < timedelta(minutes=2):
            return InferredStatus.ACTIVE
        elif age < timedelta(minutes=10):
            return InferredStatus.LIKELY_ONLINE
        else:
            return InferredStatus.POSSIBLY_OFFLINE

    @property
    def seconds_since_activity(self) -> Optional[int]:
        """Seconds since last activity, or None if never seen."""
        if not self.last_activity:
            return None
        return int((datetime.now() - self.last_activity).total_seconds())


@dataclass
class OutgoingMessage:
    """Tracks a message we sent to another agent."""
    message_id: int
    target_agent: str
    content: str
    sent_at: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    response_received: bool = False
    response_message_id: Optional[int] = None


class DistributedAgentTracker:
    """
    Local tracker for agent-to-agent communication.

    Each agent creates its own instance - there is NO shared state.
    Communication happens only through Discord messages and reactions.

    Usage:
        tracker = DistributedAgentTracker(
            own_name="canopus",
            agent_registry={"vega": 123, "polaris": 456, ...}
        )

        # When sending a message
        tracker.track_outgoing(message_id, "polaris", content)

        # When seeing activity from other agents
        tracker.observe_message(agent_name, message)
        tracker.observe_reaction(agent_name, message_id)

        # Check if an agent is likely online
        if tracker.is_agent_responsive("polaris"):
            ...
    """

    def __init__(
        self,
        own_name: str,
        agent_registry: Dict[str, int],
        ack_timeout: float = 60.0,
    ):
        """
        Initialize tracker.

        Args:
            own_name: Name of this agent
            agent_registry: Mapping of agent names to Discord user IDs
            ack_timeout: Seconds to wait for acknowledgment
        """
        self.own_name = own_name.lower()
        self.agent_registry = {k.lower(): v for k, v in agent_registry.items()}
        self.ack_timeout = ack_timeout

        # Local observations of other agents
        self._observations: Dict[str, AgentObservation] = {}

        # Messages we've sent that are awaiting acknowledgment
        self._outgoing: Dict[int, OutgoingMessage] = {}

        # Futures waiting for acknowledgment
        self._ack_waiters: Dict[int, asyncio.Future] = {}

        # Initialize observations for all known agents
        for name, user_id in self.agent_registry.items():
            if name != self.own_name:
                self._observations[name] = AgentObservation(
                    agent_name=name,
                    user_id=user_id
                )

        logger.info(
            f"[DistributedAgentTracker] Initialized for {own_name}, "
            f"tracking {len(self._observations)} other agents"
        )

    # === Observing Other Agents ===

    def observe_message(self, message: 'discord.Message') -> Optional[str]:
        """
        Record observation of a message from another agent.

        Call this in on_message for any message from a known agent.
        Returns the agent name if it was from a known agent.
        """
        author_id = message.author.id

        # Find which agent this is
        for name, user_id in self.agent_registry.items():
            if user_id == author_id and name != self.own_name:
                obs = self._observations.get(name)
                if obs:
                    obs.last_message_seen = datetime.now()

                    # Check if this is a response to one of our messages
                    if message.reference and message.reference.message_id:
                        ref_id = message.reference.message_id
                        if ref_id in self._outgoing:
                            self._handle_response(ref_id, message.id)

                    logger.debug(f"[Tracker] Observed message from {name}")
                    return name

        return None

    def observe_reaction(
        self,
        message_id: int,
        user_id: int,
        emoji: str
    ) -> Optional[str]:
        """
        Record observation of a reaction from another agent.

        Call this in on_reaction_add for reactions from known agents.
        Returns the agent name if it was from a known agent.
        """
        # Find which agent added the reaction
        for name, agent_user_id in self.agent_registry.items():
            if agent_user_id == user_id and name != self.own_name:
                obs = self._observations.get(name)
                if obs:
                    obs.last_reaction_seen = datetime.now()

                    # Check if this is an acknowledgment of our message
                    if message_id in self._outgoing:
                        if emoji in (REACTION_ACK, REACTION_DONE, REACTION_BUSY):
                            self._handle_acknowledgment(message_id, name)

                    logger.debug(f"[Tracker] Observed reaction from {name}: {emoji}")
                    return name

        return None

    def _handle_acknowledgment(self, message_id: int, agent_name: str) -> None:
        """Handle acknowledgment of a message we sent."""
        outgoing = self._outgoing.get(message_id)
        if not outgoing:
            return

        if outgoing.target_agent != agent_name:
            logger.warning(
                f"[Tracker] Ack from wrong agent: expected {outgoing.target_agent}, "
                f"got {agent_name}"
            )
            return

        outgoing.acknowledged = True
        outgoing.acknowledged_at = datetime.now()

        # Update observation
        obs = self._observations.get(agent_name)
        if obs:
            obs.acknowledged_messages.add(message_id)
            if message_id in obs.pending_messages:
                del obs.pending_messages[message_id]

        # Resolve any waiter
        if message_id in self._ack_waiters:
            future = self._ack_waiters.pop(message_id)
            if not future.done():
                future.set_result(True)

        latency = (outgoing.acknowledged_at - outgoing.sent_at).total_seconds()
        logger.info(
            f"[Tracker] Message {message_id} acknowledged by {agent_name} "
            f"(latency: {latency:.2f}s)"
        )

    def _handle_response(self, original_msg_id: int, response_msg_id: int) -> None:
        """Handle a response to one of our messages."""
        outgoing = self._outgoing.get(original_msg_id)
        if outgoing:
            outgoing.response_received = True
            outgoing.response_message_id = response_msg_id

            obs = self._observations.get(outgoing.target_agent)
            if obs:
                obs.last_mention_response = datetime.now()

            logger.info(
                f"[Tracker] Response received for message {original_msg_id} "
                f"from {outgoing.target_agent}"
            )

    # === Tracking Outgoing Messages ===

    def track_outgoing(
        self,
        message_id: int,
        target_agent: str,
        content: str
    ) -> None:
        """
        Track a message we sent to another agent.

        Call this after successfully sending a mention.
        """
        target_agent = target_agent.lower()

        outgoing = OutgoingMessage(
            message_id=message_id,
            target_agent=target_agent,
            content=content,
            sent_at=datetime.now()
        )
        self._outgoing[message_id] = outgoing

        # Also track in observation
        obs = self._observations.get(target_agent)
        if obs:
            obs.pending_messages[message_id] = datetime.now()

        logger.debug(
            f"[Tracker] Tracking outgoing message {message_id} to {target_agent}"
        )

    async def wait_for_acknowledgment(
        self,
        message_id: int,
        timeout: float = None
    ) -> bool:
        """
        Wait for acknowledgment of a message.

        Returns True if acknowledged, False if timeout.
        """
        if timeout is None:
            timeout = self.ack_timeout

        # Check if already acknowledged
        outgoing = self._outgoing.get(message_id)
        if outgoing and outgoing.acknowledged:
            return True

        # Create future to wait on
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._ack_waiters[message_id] = future

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[Tracker] Acknowledgment timeout for message {message_id}")
            return False
        finally:
            self._ack_waiters.pop(message_id, None)

    # === Status Queries ===

    def is_agent_responsive(self, agent_name: str) -> bool:
        """Check if an agent appears to be responsive based on recent activity."""
        obs = self._observations.get(agent_name.lower())
        if not obs:
            return False

        status = obs.inferred_status
        return status in (InferredStatus.ACTIVE, InferredStatus.LIKELY_ONLINE)

    def get_agent_status(self, agent_name: str) -> Optional[dict]:
        """Get status info for an agent."""
        obs = self._observations.get(agent_name.lower())
        if not obs:
            return None

        return {
            "agent_name": obs.agent_name,
            "status": obs.inferred_status.value,
            "seconds_since_activity": obs.seconds_since_activity,
            "last_activity": obs.last_activity.isoformat() if obs.last_activity else None,
            "pending_messages": len(obs.pending_messages),
            "is_responsive": self.is_agent_responsive(agent_name),
        }

    def get_all_status(self) -> Dict[str, dict]:
        """Get status for all tracked agents."""
        return {
            name: self.get_agent_status(name)
            for name in self._observations.keys()
        }

    def get_pending_messages(self, agent_name: str = None) -> List[OutgoingMessage]:
        """Get messages that haven't been acknowledged yet."""
        if agent_name:
            return [
                msg for msg in self._outgoing.values()
                if msg.target_agent == agent_name.lower() and not msg.acknowledged
            ]
        return [msg for msg in self._outgoing.values() if not msg.acknowledged]

    # === Cleanup ===

    def cleanup_old_messages(self, max_age_seconds: int = 3600) -> int:
        """Clean up old tracked messages."""
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)

        old_ids = [
            msg_id for msg_id, msg in self._outgoing.items()
            if msg.sent_at < cutoff
        ]

        for msg_id in old_ids:
            del self._outgoing[msg_id]

        return len(old_ids)


class AgentAcknowledgmentMixin:
    """
    Mixin for Discord Cogs to handle the acknowledgment protocol.

    Add this mixin to your cog to automatically:
    - Send acknowledgment reactions when mentioned
    - Track observed activity from other agents
    - Provide status checking capabilities

    Usage:
        class MyAgentCog(commands.Cog, AgentAcknowledgmentMixin):
            def __init__(self, bot):
                self.bot = bot
                self.tracker = DistributedAgentTracker(...)

            @commands.Cog.listener()
            async def on_message(self, message):
                # Track observations
                self.observe_agent_message(message)

                # Your message handling...

            @commands.Cog.listener()
            async def on_reaction_add(self, reaction, user):
                self.observe_agent_reaction(reaction, user)
    """

    # Override these in your cog
    tracker: DistributedAgentTracker
    bot: 'commands.Bot'

    def observe_agent_message(self, message: 'discord.Message') -> Optional[str]:
        """Call this in on_message to track agent activity."""
        if hasattr(self, 'tracker'):
            return self.tracker.observe_message(message)
        return None

    def observe_agent_reaction(
        self,
        reaction: 'discord.Reaction',
        user: 'discord.User'
    ) -> Optional[str]:
        """Call this in on_reaction_add to track agent reactions."""
        if hasattr(self, 'tracker'):
            return self.tracker.observe_reaction(
                message_id=reaction.message.id,
                user_id=user.id,
                emoji=str(reaction.emoji)
            )
        return None

    async def send_acknowledgment(
        self,
        message: 'discord.Message',
        status: str = "received"
    ) -> bool:
        """
        Send acknowledgment reaction to a message.

        Call this when your agent receives and starts processing a mention.

        Args:
            message: The Discord message to acknowledge
            status: One of "received", "done", "busy", "error", "seen"
        """
        emoji_map = {
            "received": REACTION_ACK,
            "done": REACTION_DONE,
            "busy": REACTION_BUSY,
            "error": REACTION_ERROR,
            "seen": REACTION_SEEN,
        }

        emoji = emoji_map.get(status, REACTION_ACK)

        try:
            await message.add_reaction(emoji)
            logger.debug(f"[AckMixin] Sent {status} acknowledgment to message {message.id}")
            return True
        except Exception as e:
            logger.warning(f"[AckMixin] Failed to add reaction: {e}")
            return False

    async def send_completion(self, message: 'discord.Message') -> bool:
        """Mark a message as fully processed."""
        return await self.send_acknowledgment(message, "done")

    def extract_source_agent(self, content: str) -> Optional[str]:
        """
        Extract the source agent name from message content.

        Messages from MentionAgentTool include a `[from: agent_name]` marker.
        """
        match = re.search(r'\[from:\s*(\w+)\]', content)
        if match:
            return match.group(1).lower()
        return None

    def is_inter_agent_message(self, message: 'discord.Message') -> bool:
        """Check if a message is from another agent in our registry."""
        if not hasattr(self, 'tracker'):
            return False

        for name, user_id in self.tracker.agent_registry.items():
            if message.author.id == user_id and name != self.tracker.own_name:
                return True
        return False
