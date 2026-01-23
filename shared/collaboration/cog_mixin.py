"""Mixin for Discord cogs to support collaborative agent behavior."""

import asyncio
import logging
import random
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

import discord

from shared.collaboration.evaluator import (
    ChimeInContext,
    ChimeInDecision,
    ChimeInResult,
)
from shared.collaboration.suppression import (
    SuppressionManager,
    parse_suppression_command,
)
from shared.base_agent import AgentContext
from shared.tools.stay_quiet import check_quiet_preferences

if TYPE_CHECKING:
    from shared.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CollaborativeCogMixin:
    """
    Mixin for Discord cogs to enable collaborative agent behavior.

    This mixin adds:
    - Chime-in evaluation for messages not directly addressed to the agent
    - Natural delay before responding to avoid all agents responding instantly
    - Cooldown to prevent infinite response loops
    - Memory-based preference checking for when to stay quiet
    - Integration with EventDispatcher for external events

    Usage:
        class MyCog(commands.Cog, CollaborativeCogMixin):
            def __init__(self, bot):
                self.bot = bot
                self.agent = MyAgent(...)
                self.main_channel_id = config.main_channel_id

            @commands.Cog.listener()
            async def on_message(self, message):
                # ... normal message handling ...

                # Add collaborative evaluation
                await self.evaluate_and_maybe_chime_in(message)
    """

    # These must be set by the implementing cog
    bot: discord.Client
    agent: 'BaseAgent'
    main_channel_id: Optional[int]
    memory_manager: Optional[object]
    _agent_registry: dict

    # Cooldown tracking - prevents infinite chime-in loops
    _last_chime_in_time: Optional[datetime] = None
    _chime_in_cooldown_seconds: int = 30  # Wait 30 seconds between chime-ins (processing takes ~10s)
    _evaluation_in_progress: bool = False  # Prevents stacking evaluations during cooldown

    async def evaluate_and_maybe_chime_in(
        self,
        message: discord.Message,
        recent_messages: Optional[List[str]] = None,
    ) -> bool:
        """
        Evaluate whether to chime in on a message and respond if appropriate.

        Args:
            message: The Discord message to evaluate
            recent_messages: Optional list of recent messages for context

        Returns:
            True if the agent decided to respond, False otherwise
        """
        # Only evaluate for main channel
        if not self.main_channel_id or message.channel.id != self.main_channel_id:
            return False

        # Never chime-in to own messages
        if message.author == self.bot.user:
            return False

        # Check session-level suppression (e.g., user said "stop testing")
        suppression = SuppressionManager.get_instance().is_suppressed(
            agent_name=self.agent.name,
            channel_id=message.channel.id,
        )
        if suppression:
            logger.debug(
                f"[{self.agent.name}] Skipping chime-in - suppressed: {suppression.reason}"
            )
            return False

        # Early exit: Check cooldown FIRST before any delays
        # This prevents "stacking" evaluations during cooldown
        now = datetime.now()
        if self._last_chime_in_time is not None:
            elapsed = (now - self._last_chime_in_time).total_seconds()
            if elapsed < self._chime_in_cooldown_seconds:
                logger.debug(
                    f"[{self.agent.name}] Skipping chime-in - "
                    f"cooldown active ({elapsed:.1f}s / {self._chime_in_cooldown_seconds}s)"
                )
                return False

        # Early exit: If another evaluation is already in progress, skip
        # This prevents multiple messages from queueing up evaluations
        if self._evaluation_in_progress:
            logger.debug(f"[{self.agent.name}] Skipping chime-in - evaluation already in progress")
            return False

        # Check if message is from another agent
        is_from_agent = False
        for agent_name, agent_id in self._agent_registry.items():
            if message.author.id == agent_id:
                is_from_agent = True
                break

        # Stagger evaluation timing to allow agents to see each other's responses
        # For user messages: longer delay (1-5s) so agents can coordinate
        # For agent messages: shorter delay (0.5-2s) to allow follow-up contributions
        if is_from_agent:
            eval_delay = random.uniform(0.5, 2.0)
        else:
            eval_delay = random.uniform(1.0, 5.0)

        logger.debug(f"[{self.agent.name}] Waiting {eval_delay:.1f}s before chime-in evaluation")
        await asyncio.sleep(eval_delay)

        # Re-check cooldown after delay (in case another agent responded during our wait)
        now = datetime.now()
        if self._last_chime_in_time is not None:
            elapsed = (now - self._last_chime_in_time).total_seconds()
            if elapsed < self._chime_in_cooldown_seconds:
                logger.debug(
                    f"[{self.agent.name}] Skipping chime-in after delay - "
                    f"cooldown now active ({elapsed:.1f}s / {self._chime_in_cooldown_seconds}s)"
                )
                return False

        # If this is an agent message, check if we've already spoken recently
        # This prevents cascading responses where agents keep responding to each other
        if is_from_agent:
            already_spoke = await self._check_already_spoke_recently(message.channel)
            if already_spoke:
                logger.debug(
                    f"[{self.agent.name}] Skipping chime-in - already spoke in recent messages"
                )
                return False

        # Skip if this agent was mentioned (will be handled normally)
        agent_name = getattr(self, 'agent', None)
        if agent_name and hasattr(agent_name, 'name'):
            my_user_id = self._agent_registry.get(agent_name.name.lower())
            if my_user_id and any(u.id == my_user_id for u in message.mentions):
                return False

        # Build chime-in context
        context = await self._build_chime_in_context(message, recent_messages)
        if not context:
            return False

        # Get cached memory context (built during last process() call)
        memory_context = self._get_chime_in_memory_context()

        # Mark evaluation in progress to prevent stacking
        self._evaluation_in_progress = True
        try:
            # Evaluate chime-in
            try:
                logger.info(f"[{self.agent.name}] Starting chime-in evaluation for: {context.message_content[:50]}...")
                result = await self.agent.evaluate_chime_in(context, memory_context)
                logger.info(f"[{self.agent.name}] Chime-in result: decision={result.decision}, confidence={result.confidence:.2f}")
            except Exception as e:
                logger.error(f"[{self.agent.name}] Chime-in evaluation error: {e}", exc_info=True)
                return False

            if result.decision == ChimeInDecision.RESPOND:
                # Add natural delay (0.5-2.5 seconds) to feel more organic
                delay = random.uniform(0.5, 2.5)
                await asyncio.sleep(delay)

                try:
                    # Show typing indicator while processing
                    async with message.channel.typing():
                        # Build agent context and process through main LLM pipeline
                        # Mark as chime-in so agent keeps response simple (no complex workflows)
                        agent_context = AgentContext(
                            channel_id=message.channel.id,
                            user_id=message.author.id,
                            message_content=message.content,
                            guild_id=message.guild.id if message.guild else None,
                            conversation_id=str(message.channel.id),
                            is_chime_in=True,  # Signal to agent: keep it simple
                        )

                        # Add chime-in reasoning as additional context
                        if result.reasoning:
                            logger.debug(f"[{self.agent.name}] Chime-in reasoning: {result.reasoning}")

                        # Process through the main agent (uses main LLM with full context)
                        response = await self.agent.process(agent_context)

                        if response.content and not response.error:
                            await message.channel.send(response.content)
                            self._last_chime_in_time = datetime.now()  # Update cooldown timer
                            logger.info(
                                f"[{self.agent.name}] Chimed in: {response.content[:50]}..."
                            )
                            return True
                        else:
                            logger.warning(f"[{self.agent.name}] Chime-in process returned no content or error")

                except discord.HTTPException as e:
                    logger.error(f"Failed to send chime-in: {e}")
                except Exception as e:
                    logger.error(f"[{self.agent.name}] Chime-in processing error: {e}", exc_info=True)

            elif result.decision == ChimeInDecision.ACKNOWLEDGE_ONLY:
                # Just add a reaction, no text response needed
                reaction = result.suggested_reaction or "\u2705"  # Default: checkmark
                try:
                    await message.add_reaction(reaction)
                    self._last_chime_in_time = datetime.now()
                    logger.info(f"[{self.agent.name}] Acknowledged with reaction: {reaction}")
                    return True
                except discord.HTTPException as e:
                    logger.warning(f"[{self.agent.name}] Failed to add reaction: {e}")

            return False
        finally:
            # Always clear the flag when done
            self._evaluation_in_progress = False

    async def _build_chime_in_context(
        self,
        message: discord.Message,
        recent_messages: Optional[List[str]] = None,
    ) -> Optional[ChimeInContext]:
        """Build ChimeInContext from a Discord message."""
        # Determine if from another agent
        is_from_agent = False
        for agent_name, agent_id in self._agent_registry.items():
            if message.author.id == agent_id:
                is_from_agent = True
                break

        # Get mentioned agents
        mentioned_agents = []
        for user in message.mentions:
            for agent_name, agent_id in self._agent_registry.items():
                if user.id == agent_id:
                    mentioned_agents.append(agent_name)

        # Get recent messages for context if not provided
        # Fetch more messages (10) to capture recent agent responses
        if recent_messages is None:
            recent_messages = await self._fetch_recent_messages(
                message.channel, limit=10
            )

        return ChimeInContext(
            channel_id=message.channel.id,
            message_content=message.content,
            message_author=message.author.display_name,
            message_author_id=message.author.id,
            timestamp=datetime.now(),
            is_from_agent=is_from_agent,
            mentioned_agents=mentioned_agents,
            recent_messages=recent_messages,
        )

    async def _fetch_recent_messages(
        self,
        channel: discord.TextChannel,
        limit: int = 5,
    ) -> List[str]:
        """Fetch recent messages from a channel for context."""
        messages = []
        try:
            async for msg in channel.history(limit=limit + 1):  # +1 to skip current
                if len(messages) >= limit:
                    break
                # Format: "Author: content"
                messages.append(f"{msg.author.display_name}: {msg.content[:100]}")
        except discord.HTTPException:
            pass
        return list(reversed(messages))  # Oldest first

    async def _check_already_spoke_recently(
        self,
        channel: discord.TextChannel,
        lookback: int = 10,
    ) -> bool:
        """
        Check if this agent has already spoken in recent channel history.

        This prevents agents from responding to each other's messages
        when they've already participated in the conversation burst.

        Args:
            channel: The Discord channel to check
            lookback: Number of recent messages to check

        Returns:
            True if this agent has spoken in recent messages
        """
        my_user_id = self._agent_registry.get(self.agent.name.lower())
        if not my_user_id:
            return False

        try:
            async for msg in channel.history(limit=lookback):
                if msg.author.id == my_user_id:
                    logger.debug(
                        f"[{self.agent.name}] Found own message in recent history: "
                        f"'{msg.content[:30]}...'"
                    )
                    return True
        except discord.HTTPException:
            pass

        return False

    def _get_chime_in_memory_context(self) -> str:
        """
        Get cached memory context for chime-in evaluation.

        Uses the memory context that was cached during the last process() call.
        This includes the agent's full persona:
        - Short-term memories (recent interactions, active context)
        - Category summaries (what the agent knows about)
        - Auto-selected relevant long-term memories

        This avoids duplicate Redis calls - the main process builds and caches it,
        chime-in reuses the cache.
        """
        if not hasattr(self, 'agent') or not self.agent:
            return ""

        # Use the cached memory context from the agent
        return self.agent.get_cached_memory_context()

    async def check_and_handle_suppression_command(
        self,
        message: discord.Message,
    ) -> bool:
        """
        Check if a message contains a suppression command and handle it.

        This should be called early in message processing to detect commands
        like "@Canopus stop testing" or "everyone stand down".

        Args:
            message: The Discord message to check

        Returns:
            True if a suppression command was handled (caller may want to skip further processing)
        """
        if not hasattr(self, '_agent_registry') or not self._agent_registry:
            return False

        parsed = parse_suppression_command(message.content, self._agent_registry)
        if not parsed:
            return False

        suppression_mgr = SuppressionManager.get_instance()

        if parsed["action"] == "suppress":
            target_agent = parsed["agent"]

            # Check if this suppression is directed at this agent (or everyone)
            if target_agent == "*" or target_agent == self.agent.name.lower():
                suppression_mgr.add_suppression(
                    agent_name=target_agent,
                    reason=f"User requested: {message.content[:50]}",
                    duration_seconds=parsed.get("duration", 600),
                    channel_id=message.channel.id,
                    activity_pattern=parsed.get("activity"),
                )

                # Only acknowledge if this agent is specifically targeted
                if target_agent == self.agent.name.lower():
                    try:
                        await message.add_reaction("\U0001F64A")  # speak-no-evil monkey
                    except discord.HTTPException:
                        pass

                logger.info(
                    f"[{self.agent.name}] Suppression activated by user command: {message.content[:50]}"
                )
                return True

        elif parsed["action"] == "resume":
            target_agent = parsed["agent"]

            if target_agent == self.agent.name.lower():
                cleared = suppression_mgr.clear_suppression(
                    agent_name=target_agent,
                    channel_id=message.channel.id,
                )

                if cleared:
                    try:
                        await message.add_reaction("\U0001F44B")  # wave
                    except discord.HTTPException:
                        pass

                    logger.info(f"[{self.agent.name}] Suppression cleared by user command")
                return True

        return False

    async def handle_event_notification(
        self,
        event_type: str,
        description: str,
        payload: dict,
    ):
        """
        Handle an event notification from the EventDispatcher.

        This method is called when an event is routed to this agent.
        Override in subclass for custom event handling.

        Args:
            event_type: Type of event (e.g., "session.crashed")
            description: Human-readable description
            payload: Event-specific data
        """
        logger.info(
            f"[{self.agent.name}] Received event: {event_type} - {description}"
        )
        # Default: no action, subclasses can override


def create_event_handler(cog: CollaborativeCogMixin):
    """
    Create an async event handler function for the EventDispatcher.

    Args:
        cog: The cog that will handle events

    Returns:
        Async function that can be registered with EventDispatcher
    """
    async def handler(event):
        await cog.handle_event_notification(
            event_type=event.event_type,
            description=event.get_description(),
            payload=event.payload,
        )

    return handler
