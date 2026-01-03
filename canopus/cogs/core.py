"""Canopus Core Cog - Main Discord integration for Canopus bot."""

import asyncio
import logging
from typing import Optional

import discord
from discord.ext import commands, tasks

from shared.base_agent import AgentContext
from shared.agent_messaging import AgentMessaging
from shared.agent_queue import AgentMessageQueue
from shared.workflow_state import QueueWorkflowStateProvider
from shared.agent_coordinator import (
    DistributedAgentTracker,
    AgentAcknowledgmentMixin,
    REACTION_ACK,
    REACTION_DONE,
)
from shared.collaboration import CollaborativeCogMixin
from canopus.agent import CanopusAgent
from canopus.browser.manager import BrowserSessionManager
from canopus.config import get_config

logger = logging.getLogger(__name__)


class CanopusCore(commands.Cog, CollaborativeCogMixin, AgentAcknowledgmentMixin):
    """
    Core cog for the Canopus Discord bot.

    Handles:
    - Message routing to CanopusAgent when mentioned
    - Inter-agent communication via @mentions
    - Browser session management per channel
    - Cleanup of idle browser sessions
    - Distributed agent tracking and acknowledgment
    - Collaborative chime-in evaluation for proactive agent participation
    """

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = get_config()

        # Components initialized in cog_load
        self.agent: Optional[CanopusAgent] = None
        self.browser_manager: Optional[BrowserSessionManager] = None
        self.agent_messaging: Optional[AgentMessaging] = None
        self.conversation_manager = None
        self.memory_manager = None

        # Workflow state provider for context awareness
        self.workflow_state_provider = QueueWorkflowStateProvider("Canopus")

        # Message queue for sequential processing
        self.message_queue = AgentMessageQueue(
            agent_name="Canopus",
            process_callback=self._process_queued_message,
        )

        # Connect workflow state provider to the queue
        self.workflow_state_provider.set_queue(self.message_queue)

        # Distributed tracker for inter-agent communication
        # Each agent has its own tracker - no shared state
        self.tracker: Optional[DistributedAgentTracker] = None

        # Track message IDs we've acknowledged to avoid duplicate acks
        self._acknowledged_message_ids: set = set()

        # CollaborativeCogMixin requirements
        self.main_channel_id: Optional[int] = self.config.main_channel_id if hasattr(self.config, 'main_channel_id') else None
        self._agent_registry: dict = self.config.agent_registry.agents if self.config.agent_registry else {}

    async def cog_load(self):
        """Async initialization after cog is loaded."""
        logger.info("Initializing Canopus Core cog...")

        # Initialize components
        await self._init_components()

        # Initialize distributed tracker for inter-agent communication
        self.tracker = DistributedAgentTracker(
            own_name="canopus",
            agent_registry=self.config.agent_registry.agents,
        )
        logger.info("Distributed agent tracker initialized")

        # Start the message queue processor
        self.message_queue.start()

        # Start cleanup loop
        self.cleanup_idle_sessions.start()

        logger.info("Canopus Core cog initialized successfully")

    def cog_unload(self):
        """Clean up when cog is unloaded."""
        self.cleanup_idle_sessions.cancel()
        self.message_queue.stop()
        # Close all browser sessions
        asyncio.create_task(self.browser_manager.close_all())

    async def _init_components(self):
        """Initialize all components."""
        # Initialize browser session manager
        self.browser_manager = BrowserSessionManager(
            browser_type=self.config.browser_type,
            headless=self.config.headless,
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height,
            max_sessions_per_channel=self.config.max_sessions_per_channel,
            session_idle_timeout=self.config.session_idle_timeout,
            screenshot_dir=self.config.screenshot_dir,
        )
        logger.info(f"Browser manager initialized: {self.config.browser_type}, headless={self.config.headless}")

        # Initialize agent messaging
        self.agent_messaging = AgentMessaging(
            bot=self.bot,
            agent_registry=self.config.agent_registry.agents,
            own_agent_name="canopus",
        )

        # Initialize Redis-backed managers if available
        if self.config.redis_url:
            try:
                from shared.memory import MemoryManager
                from shared.conversation import ConversationManager
                self.memory_manager = MemoryManager(self.config.redis_url)
                self.conversation_manager = ConversationManager(self.config.redis_url)
                logger.info("Memory and conversation managers initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis-backed managers: {e}")

        # Initialize LLM providers
        from shared.llm.factory import create_llm_provider, create_utility_llm_from_config
        llm = create_llm_provider(
            provider=self.config.llm_provider,
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
        )

        # Create utility LLM for cheap/fast evaluations (chime-in, etc.)
        utility_llm = create_utility_llm_from_config(self.config)
        if utility_llm:
            logger.info("Initialized utility LLM for quick evaluations")
        else:
            logger.info("No utility LLM configured, using main LLM for evaluations")

        # Initialize the Canopus agent
        self.agent = CanopusAgent(
            llm=llm,
            browser_manager=self.browser_manager,
            conversation_manager=self.conversation_manager,
            memory_manager=self.memory_manager,
            discord_bot=self.bot,
            agent_registry=self.config.agent_registry.agents,
            utility_llm=utility_llm,
        )

        # Connect workflow state provider to agent
        self.agent.set_workflow_state_provider(self.workflow_state_provider)

        logger.info("Canopus agent initialized")

    @tasks.loop(minutes=5.0)
    async def cleanup_idle_sessions(self):
        """Periodic cleanup of idle browser sessions."""
        if self.browser_manager:
            await self.browser_manager.cleanup_idle_sessions()

    @cleanup_idle_sessions.before_loop
    async def before_cleanup(self):
        await self.bot.wait_until_ready()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle incoming messages."""
        # Ignore own messages
        if message.author == self.bot.user:
            return

        # Track activity from other agents (for distributed status inference)
        if self.tracker:
            observed_agent = self.tracker.observe_message(message)
            if observed_agent:
                logger.debug(f"[Canopus] Observed activity from {observed_agent}")

        # Check if from allowed user or another agent
        is_allowed_user = message.author.id == self.config.allowed_user_id
        is_from_agent = self.agent_messaging.is_from_agent(message) is not None

        if not is_allowed_user and not is_from_agent:
            return

        # Check if this bot is mentioned
        if not self.bot.user or not self.bot.user.mentioned_in(message):
            # Not mentioned - evaluate chime-in for messages in main channel
            if self.main_channel_id and message.channel.id == self.main_channel_id:
                if self.agent:
                    logger.debug(f"[Canopus] Checking chime-in for: {message.content[:50]}...")
                    await self.evaluate_and_maybe_chime_in(message)
                else:
                    logger.warning(f"[Canopus] Agent not initialized, skipping chime-in")
            return

        # Process the mention
        await self._handle_mention(message, is_from_agent)

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
                    f"[Canopus] Observed reaction from {observed_agent}: {reaction.emoji}"
                )

    async def _get_or_create_conversation(
        self,
        channel_id: int,
        user_id: int,
        guild_id: int = None
    ) -> Optional[str]:
        """Get existing conversation for channel or create a new one."""
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

    async def _handle_mention(self, message: discord.Message, is_from_agent: bool):
        """Handle when Canopus is @mentioned - queue the message for processing."""
        # Extract message content without the mention
        clean_content = self.agent_messaging.extract_message_for_agent(
            message, "canopus"
        )

        if not clean_content:
            await message.channel.send(
                "You mentioned me but didn't say anything. What would you like me to browse?"
            )
            return

        # Log who we received message from
        if is_from_agent:
            sender_agent = self.agent_messaging.is_from_agent(message)
            logger.info(f"Canopus received message from agent: {sender_agent}")

            # Send acknowledgment reaction to let sender know we received it
            # This is visible to all agents and confirms delivery
            if message.id not in self._acknowledged_message_ids:
                try:
                    await message.add_reaction(REACTION_ACK)
                    self._acknowledged_message_ids.add(message.id)
                    logger.info(f"[Canopus] Sent ✅ acknowledgment for message {message.id}")

                    # Keep set from growing indefinitely
                    if len(self._acknowledged_message_ids) > 500:
                        # Remove oldest entries
                        self._acknowledged_message_ids = set(
                            list(self._acknowledged_message_ids)[-250:]
                        )
                except Exception as e:
                    logger.warning(f"[Canopus] Failed to send acknowledgment: {e}")
        else:
            logger.info(f"Canopus received message from user: {message.author}")

        # Get or create conversation for this channel
        conversation_id = await self._get_or_create_conversation(
            channel_id=message.channel.id,
            user_id=message.author.id,
            guild_id=message.guild.id if message.guild else None
        )

        # Create agent context
        context = AgentContext(
            channel_id=message.channel.id,
            user_id=message.author.id,
            message_content=clean_content,
            guild_id=message.guild.id if message.guild else None,
            mentioned_agent="canopus",
            conversation_id=conversation_id,
        )

        # Add to queue for sequential processing
        # No hardcoded acknowledgments - the LLM handles all responses
        success, status, recent_task = await self.message_queue.enqueue(
            message=message,
            context=context,
            is_from_agent=is_from_agent,
        )

        # Handle duplicate task detection (content-based deduplication)
        if status == "duplicate_task":
            # Silent acknowledgment - just add reaction, no channel spam
            try:
                await message.add_reaction("✅")
                logger.info(
                    f"[Canopus] Duplicate task detected from {self.agent_messaging.is_from_agent(message)}, "
                    f"acknowledged silently (hash match or semantic duplicate)"
                )
            except Exception as e:
                logger.warning(f"[Canopus] Failed to add duplicate ack reaction: {e}")
            return

        if not success and status not in ("duplicate_id", "duplicate_task"):
            await message.channel.send("I'm overwhelmed right now. Please try again shortly.")

    async def _process_queued_message(
        self,
        message: discord.Message,
        context: AgentContext,
        is_from_agent: bool,
    ):
        """Process a message from the queue - this is the actual work."""
        logger.info(
            f"[Canopus] PROCESSING: msg_id={message.id}, from_agent={is_from_agent}, "
            f"content={context.message_content[:50]}..."
        )

        # Show typing indicator while processing
        async with message.channel.typing():
            response = await self.agent.process(context)

        # Track completed task for workflow continuity
        self.workflow_state_provider.on_task_completed(context.message_content)

        # Send response only if there's content
        if response.content:
            logger.info(
                f"[Canopus] SENDING RESPONSE: msg_id={message.id}, "
                f"response_len={len(response.content)}, tool_calls={response.tool_calls_made}, "
                f"content_preview={response.content[:100]}..."
            )
            await self._send_response(message.channel, response.content)
        else:
            logger.info(
                f"[Canopus] NO RESPONSE CONTENT: msg_id={message.id}, "
                f"tool_calls={response.tool_calls_made}"
            )

        # Log processing stats
        logger.info(
            f"[Canopus] PROCESS COMPLETE: msg_id={message.id}, "
            f"time={response.processing_time_ms}ms, tools={response.tool_calls_made}"
        )

        # Send completion reaction for inter-agent messages
        # This tells other agents we've finished processing
        if is_from_agent:
            try:
                await message.add_reaction(REACTION_DONE)
                logger.info(f"[Canopus] Sent ✔️ completion for message {message.id}")
            except Exception as e:
                logger.warning(f"[Canopus] Failed to send completion reaction: {e}")

    async def _send_response(self, channel: discord.TextChannel, content: str):
        """Send a response, splitting if necessary."""
        max_length = 1900
        max_total_length = 6000  # Absolute max to prevent spam

        # Detect and truncate repetitive content
        content = self._clean_repetitive_content(content)

        # Enforce total length limit
        if len(content) > max_total_length:
            content = content[:max_total_length] + "\n\n... [Response truncated due to length]"

        if len(content) <= max_length:
            await channel.send(content)
            return

        # Split into chunks (max 3 to prevent spam)
        chunks = []
        remaining = content
        while remaining and len(chunks) < 3:
            if len(remaining) <= max_length:
                chunks.append(remaining)
                break

            # Find a good split point
            split_at = max_length
            for sep in ['\n', ' ']:
                idx = remaining.rfind(sep, 0, max_length)
                if idx > max_length // 2:
                    split_at = idx + 1
                    break

            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]

        # If there's more content, add truncation notice
        if remaining:
            chunks[-1] = chunks[-1].rstrip() + "\n\n... [Response truncated]"

        for chunk in chunks:
            await channel.send(chunk)

    def _clean_repetitive_content(self, content: str) -> str:
        """Detect and clean repetitive content patterns."""
        lines = content.split('\n')

        # Detect repetitive lines
        seen_lines = {}
        cleaned_lines = []
        repeat_count = 0

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                cleaned_lines.append(line)
                continue

            # Track line occurrences
            if line_stripped in seen_lines:
                seen_lines[line_stripped] += 1
                if seen_lines[line_stripped] <= 2:
                    cleaned_lines.append(line)
                else:
                    repeat_count += 1
            else:
                seen_lines[line_stripped] = 1
                cleaned_lines.append(line)

        result = '\n'.join(cleaned_lines)

        # If we removed many repeats, note it
        if repeat_count > 5:
            result += f"\n\n[Removed {repeat_count} repetitive lines]"

        return result

    # === Commands ===

    @commands.command(name="browse")
    async def cmd_browse(self, ctx: commands.Context, url: str):
        """Quick navigation to a URL."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        try:
            session = await self.browser_manager.get_or_create(ctx.channel.id)
            result = await session.navigate(url)

            if result.get("success"):
                await ctx.send(f"🌐 Navigated to: {result['url']}\nTitle: {result.get('title', 'N/A')}")
            else:
                await ctx.send(f"❌ Navigation failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

    @commands.command(name="screenshot")
    async def cmd_screenshot(self, ctx: commands.Context):
        """Take a screenshot of the current page."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        session = await self.browser_manager.get(ctx.channel.id)
        if not session:
            await ctx.send("No active browser session. Use `!browse <url>` first.")
            return

        try:
            from datetime import datetime
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = self.browser_manager.get_screenshot_path(filename)

            result = await session.screenshot(path=str(path))

            if result.get("success"):
                file = discord.File(str(path), filename=filename)
                await ctx.send(f"📸 Screenshot of {session.current_url}", file=file)
            else:
                await ctx.send(f"❌ Screenshot failed: {result.get('error')}")
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

    @commands.command(name="browser-status")
    async def cmd_browser_status(self, ctx: commands.Context):
        """Show browser session status."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        sessions = self.browser_manager.list_sessions()
        if not sessions:
            await ctx.send("No active browser sessions.")
            return

        lines = ["**Active Browser Sessions:**"]
        for s in sessions:
            url = s.get("current_url", "about:blank")
            if len(url) > 60:
                url = url[:60] + "..."
            lines.append(f"- {s['session_id']}: {url}")

        await ctx.send("\n".join(lines))

    @commands.command(name="close-browser")
    async def cmd_close_browser(self, ctx: commands.Context):
        """Close the browser session for this channel."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        closed = await self.browser_manager.close(ctx.channel.id)
        if closed:
            await ctx.send("🔒 Browser session closed.")
        else:
            await ctx.send("No active browser session to close.")


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(CanopusCore(bot))
