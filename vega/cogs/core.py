"""Vega Core Cog - Conversational Discord bot functionality."""

import asyncio
import logging
import os
from typing import Optional

import discord
from discord.ext import commands

from vega.config import Config
from vega.agents import VegaAgent, AgentContext
from shared.agent_queue import AgentMessageQueue
from shared.collaboration import CollaborativeCogMixin
from shared.agent_coordinator import (
    DistributedAgentTracker,
    AgentAcknowledgmentMixin,
    REACTION_ACK,
    REACTION_DONE,
)

logger = logging.getLogger(__name__)


class VegaCore(commands.Cog, CollaborativeCogMixin, AgentAcknowledgmentMixin):
    """
    Vega Discord Bot - Head conversational agent.

    Vega is a pure conversational agent. It handles natural language
    conversation and delegates hands-on work to Altair via @mentions.

    All commands (!vp, !vli, etc.) are handled by Altair.
    Supports distributed agent tracking and acknowledgment via Discord reactions.
    """

    def __init__(self, bot):
        self.bot = bot

        # Agent components
        self.vega_agent = None
        self.llm = None
        self.project_manager = None
        self.conversation_manager = None
        self.memory_manager = None
        self._agent_initialized = False
        self._agent_registry = {}  # Maps agent names to Discord user IDs

        # Main channel for collaborative agent communication (uses AGENT_CHANNEL_ID)
        self.main_channel_id = Config.AGENT_CHANNEL_ID if hasattr(Config, 'AGENT_CHANNEL_ID') else None

        # Distributed tracker for inter-agent communication
        # Each agent has its own tracker - no shared state
        self.tracker: Optional[DistributedAgentTracker] = None

        # Track message IDs we've acknowledged to avoid duplicate acks
        self._acknowledged_message_ids: set = set()

        # Message queue for sequential processing
        self.message_queue = AgentMessageQueue(
            agent_name="Vega",
            process_callback=self._process_queued_message,
        )

        # Initialize agent
        self._init_agent()

        # Start message queue
        self.message_queue.start()

    @property
    def agent(self):
        """Get the agent instance (required by CollaborativeCogMixin)."""
        return self.vega_agent

    def _load_agent_registry(self) -> dict:
        """Load agent registry from environment variables."""
        registry = {}

        for key, value in os.environ.items():
            if key.startswith("AGENT_") and key.endswith("_USER_ID"):
                agent_name = key[6:-8].lower()
                try:
                    registry[agent_name] = int(value)
                except ValueError:
                    logger.warning(f"Invalid user ID for {agent_name}: {value}")

        if registry:
            logger.info(f"Loaded agent registry: {list(registry.keys())}")
        else:
            logger.warning("No agent registry configured (set AGENT_*_USER_ID env vars)")

        return registry

    def _init_agent(self):
        """Initialize Vega agent."""
        if not Config.LLM_API_KEY:
            logger.info("LLM_API_KEY not configured, agent features disabled")
            return

        try:
            from shared.llm.factory import create_llm_provider, create_utility_llm_from_config
            from vega.config import get_config

            config = get_config()

            # Create LLM provider
            self.llm = create_llm_provider(
                provider=Config.LLM_PROVIDER,
                api_key=Config.LLM_API_KEY,
                model=Config.LLM_MODEL
            )
            logger.info(f"Initialized {Config.LLM_PROVIDER} LLM provider")

            # Create utility LLM for cheap/fast evaluations (chime-in, etc.)
            self.utility_llm = create_utility_llm_from_config(config)
            if self.utility_llm:
                logger.info(f"Initialized utility LLM for quick evaluations")
            else:
                logger.info("No utility LLM configured, using main LLM for evaluations")

            # Initialize Redis-backed managers if available
            if Config.REDIS_URL:
                try:
                    from shared.project import ProjectManager
                    from shared.conversation import ConversationManager
                    from shared.memory import MemoryManager

                    self.project_manager = ProjectManager(Config.REDIS_URL)
                    self.conversation_manager = ConversationManager(Config.REDIS_URL)
                    self.memory_manager = MemoryManager(Config.REDIS_URL)
                    logger.info("Initialized Redis-backed managers (including MemoryManager)")
                except ImportError as e:
                    logger.warning(f"Redis managers not available: {e}")

            # Load agent registry for @mention support
            self._agent_registry = self._load_agent_registry()

            # Initialize distributed tracker for inter-agent communication
            if self._agent_registry:
                self.tracker = DistributedAgentTracker(
                    own_name="vega",
                    agent_registry=self._agent_registry,
                )
                logger.info("Distributed agent tracker initialized")

            # Create Vega agent
            self.vega_agent = VegaAgent(
                llm=self.llm,
                project_manager=self.project_manager,
                conversation_manager=self.conversation_manager,
                memory_manager=self.memory_manager,
                discord_bot=self.bot,
                agent_registry=self._agent_registry,
                utility_llm=self.utility_llm,
            )

            self._agent_initialized = True
            logger.info("VegaAgent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}", exc_info=True)

    def cog_unload(self):
        self.message_queue.stop()

    # --------------------------------------------------
    # MESSAGE HANDLER
    # --------------------------------------------------

    def _get_mentioned_other_agent(self, message: discord.Message) -> str | None:
        """
        Check if another agent (not Vega) was mentioned in the message.

        Returns the agent name if found, None otherwise.
        """
        for user in message.mentions:
            # Check if this mentioned user is a known agent (and not Vega)
            for agent_name, agent_id in self._agent_registry.items():
                if user.id == agent_id and agent_name != 'vega':
                    return agent_name

        return None

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        # Track activity from other agents (for distributed status inference)
        if self.tracker:
            observed_agent = self.tracker.observe_message(message)
            if observed_agent:
                logger.debug(f"[Vega] Observed activity from {observed_agent}")

        if not Config.ALLOWED_USER_ID:
            await message.channel.send(
                f"**Setup Mode**\nYour Discord User ID is: `{message.author.id}`\n"
                f"Add `ALLOWED_USER_ID={message.author.id}` to `.env` and restart."
            )
            return

        if message.author.id != Config.ALLOWED_USER_ID:
            # Still process if it's from another agent (for chime-in)
            is_from_agent = any(
                message.author.id == agent_id
                for agent_id in self._agent_registry.values()
            )
            if not is_from_agent:
                return

        content = message.content.strip()

        # Ignore all commands - Altair handles them
        if content.startswith('!'):
            return

        # Check if user mentioned another agent - evaluate chime-in only
        mentioned_other = self._get_mentioned_other_agent(message)
        if mentioned_other:
            logger.debug(f"User mentioned @{mentioned_other}, checking chime-in for Vega")
            if self._agent_initialized and self.main_channel_id:
                await self.evaluate_and_maybe_chime_in(message)
            return

        # Check if message is from another agent - evaluate chime-in
        is_from_agent = any(
            message.author.id == agent_id
            for agent_id in self._agent_registry.values()
        )
        if is_from_agent:
            logger.debug(f"Message from agent, evaluating chime-in for Vega")
            if self._agent_initialized and self.main_channel_id:
                await self.evaluate_and_maybe_chime_in(message)
            return

        # Regular conversation: queue for Vega processing
        if content and self._agent_initialized:
            # Get or create conversation for this channel
            conversation_id = await self._get_or_create_conversation(
                channel_id=message.channel.id,
                user_id=message.author.id,
                guild_id=message.guild.id if message.guild else None
            )

            context = AgentContext(
                channel_id=message.channel.id,
                user_id=message.author.id,
                message_content=content,
                guild_id=message.guild.id if message.guild else None,
                conversation_id=conversation_id,
            )

            # Add to queue for sequential processing
            await self.message_queue.enqueue(
                message=message,
                context=context,
                is_from_agent=False,
            )

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        """Track reactions from other agents for acknowledgment detection."""
        # Ignore own reactions
        if user == self.bot.user:
            return

        # Track reactions from other agents
        if self.tracker:
            observed_agent = self.tracker.observe_reaction(
                message_id=reaction.message.id,
                user_id=user.id,
                emoji=str(reaction.emoji)
            )
            if observed_agent:
                logger.debug(
                    f"[Vega] Observed reaction from {observed_agent}: {reaction.emoji}"
                )

    async def _get_or_create_conversation(
        self,
        channel_id: int,
        user_id: int,
        guild_id: int = None
    ) -> str:
        """
        Get existing conversation for channel or create a new one.

        This ensures chat history is preserved across messages in the same channel.
        """
        if not self.conversation_manager:
            return None

        # Try to get existing conversation for this channel
        conversation_id = await self.conversation_manager.get_channel_conversation(channel_id)

        if not conversation_id:
            # Create new conversation for this channel
            conversation_id = await self.conversation_manager.create(
                user_id=user_id,
                channel_id=channel_id,
                metadata={"guild_id": guild_id}
            )
            logger.info(f"Created new conversation {conversation_id} for channel {channel_id}")

        return conversation_id

    async def _process_queued_message(
        self,
        message: discord.Message,
        context: AgentContext,
        is_from_agent: bool,
    ):
        """Process a message from the queue - this is the actual work."""
        if not self.vega_agent:
            await message.channel.send("Vega agent not available.")
            return

        async with message.channel.typing():
            try:
                response = await self.vega_agent.process(context)

                if response.content:
                    chunks = self._split_message(response.content)
                    logger.info(
                        f"[Vega] SENDING RESPONSE: msg_id={message.id}, "
                        f"chunks={len(chunks)}, total_len={len(response.content)}, "
                        f"tool_calls={response.tool_calls_made}, "
                        f"content_preview={response.content[:100]}..."
                    )
                    for i, chunk in enumerate(chunks):
                        sent = await message.channel.send(chunk)
                        logger.debug(f"[Vega] SENT CHUNK {i+1}/{len(chunks)}: sent_id={sent.id}")
                else:
                    logger.info(
                        f"[Vega] NO RESPONSE CONTENT: msg_id={message.id}, "
                        f"tool_calls={response.tool_calls_made} (tool-only response)"
                    )

                logger.info(
                    f"[Vega] PROCESS COMPLETE: msg_id={message.id}, "
                    f"time={response.processing_time_ms}ms, tools={response.tool_calls_made}"
                )

            except Exception as e:
                logger.error(f"[Vega] PROCESS ERROR: msg_id={message.id}, error={e}", exc_info=True)
                await message.channel.send(f"**Error:** {e}")

    def _split_message(self, text: str, max_len: int = 1900) -> list:
        """Split text into chunks for Discord messages."""
        if len(text) <= max_len:
            return [text]

        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break

            # Find a good split point (newline)
            split_at = max_len
            newline_pos = text.rfind('\n', 0, max_len)
            if newline_pos > max_len // 2:
                split_at = newline_pos + 1

            chunks.append(text[:split_at])
            text = text[split_at:]

        return chunks


async def setup(bot):
    await bot.add_cog(VegaCore(bot))
