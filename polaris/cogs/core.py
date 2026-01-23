"""Polaris Core Cog - Main Discord integration for Polaris bot."""

import logging
from typing import Optional

import discord
from discord.ext import commands

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
from polaris.agent import PolarisAgent
from polaris.config import get_config

logger = logging.getLogger(__name__)


class PolarisCore(commands.Cog, AgentAcknowledgmentMixin):
    """
    Core cog for the Polaris Discord bot.

    Handles:
    - Message routing to PolarisAgent when mentioned
    - Inter-agent communication via @mentions
    - Google Calendar API initialization
    - Distributed agent tracking and acknowledgment
    """

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = get_config()

        # Components initialized in cog_load
        self.agent: Optional[PolarisAgent] = None
        self.agent_messaging: Optional[AgentMessaging] = None
        self.conversation_manager = None
        self.memory_manager = None

        # Google Calendar service
        self._calendar_service = None

        # Workflow state provider for context awareness
        self.workflow_state_provider = QueueWorkflowStateProvider("Polaris")

        # Message queue for sequential processing
        self.message_queue = AgentMessageQueue(
            agent_name="Polaris",
            process_callback=self._process_queued_message,
        )

        # Connect workflow state provider to the queue
        self.workflow_state_provider.set_queue(self.message_queue)

        # Distributed tracker for inter-agent communication
        # Each agent has its own tracker - no shared state
        self.tracker: Optional[DistributedAgentTracker] = None

        # Track message IDs we've acknowledged to avoid duplicate acks
        self._acknowledged_message_ids: set = set()

    async def cog_load(self):
        """Async initialization after cog is loaded."""
        logger.info("Initializing Polaris Core cog...")

        await self._init_components()

        # Initialize distributed tracker for inter-agent communication
        self.tracker = DistributedAgentTracker(
            own_name="polaris",
            agent_registry=self.config.agent_registry.agents,
        )
        logger.info("Distributed agent tracker initialized")

        # Start message queue
        self.message_queue.start()

        logger.info("Polaris Core cog initialized successfully")

    def cog_unload(self):
        """Clean up when cog is unloaded."""
        self.message_queue.stop()

    async def _init_components(self):
        """Initialize all components."""
        # Initialize agent messaging
        self.agent_messaging = AgentMessaging(
            bot=self.bot,
            agent_registry=self.config.agent_registry.agents,
            own_agent_name="polaris",
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

        # Initialize Google Calendar service if credentials are available
        await self._init_calendar_service()

        # Initialize LLM providers
        from shared.llm.factory import create_llm_provider, create_utility_llm_from_config

        llm = create_llm_provider(
            provider=self.config.llm_provider,
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
        )

        # Create utility LLM for lightweight evaluations
        utility_llm = create_utility_llm_from_config(self.config)

        # Initialize the Polaris agent
        self.agent = PolarisAgent(
            llm=llm,
            conversation_manager=self.conversation_manager,
            memory_manager=self.memory_manager,
            discord_bot=self.bot,
            agent_registry=self.config.agent_registry.agents,
            config=self.config,
            utility_llm=utility_llm,
        )

        # Set calendar service if available
        if self._calendar_service:
            self.agent.set_calendar_service(self._calendar_service)

        # Connect workflow state provider to agent
        self.agent.set_workflow_state_provider(self.workflow_state_provider)

    async def _init_calendar_service(self):
        """Initialize Google Calendar API service if already authenticated."""
        from polaris.tools.utils import check_calendar_auth_status, get_default_token_path

        if not self.config.google_credentials_path:
            logger.warning(
                "GOOGLE_CREDENTIALS_PATH not set - calendar features will be limited"
            )
            return

        # Check if we already have valid tokens (don't trigger OAuth on startup)
        token_path = self.config.google_token_path or get_default_token_path()
        auth_status = check_calendar_auth_status(
            credentials_path=self.config.google_credentials_path,
            token_path=token_path,
        )

        if not auth_status['authenticated']:
            logger.warning(f"Calendar not authenticated: {auth_status['message']}")
            logger.info("Run !pauth in Discord to authenticate with Google Calendar")
            return

        # We have valid tokens, load the service
        try:
            from polaris.tools.utils import load_calendar_service

            self._calendar_service = load_calendar_service(
                token_path=token_path,
            )
            logger.info("Google Calendar service initialized from existing token")
        except Exception as e:
            logger.error(f"Failed to initialize Google Calendar service: {e}")
            logger.info("Run !pauth in Discord to re-authenticate")

    async def _get_or_create_conversation(
        self,
        channel_id: int,
        user_id: int,
        guild_id: int = None,
    ) -> Optional[str]:
        """Get existing conversation for channel or create a new one."""
        if not self.conversation_manager:
            return None

        conversation_id = await self.conversation_manager.get_channel_conversation(
            channel_id
        )

        if not conversation_id:
            conversation_id = await self.conversation_manager.create(
                user_id=user_id,
                channel_id=channel_id,
                metadata={"guild_id": guild_id},
            )
            logger.info(
                f"Created new conversation {conversation_id} for channel {channel_id}"
            )

        return conversation_id

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
                logger.debug(f"[Polaris] Observed activity from {observed_agent}")

        # Check if from allowed user or another agent
        is_allowed_user = message.author.id == self.config.allowed_user_id
        is_from_agent = self.agent_messaging.is_from_agent(message) is not None

        if not is_allowed_user and not is_from_agent:
            return

        # Only respond when explicitly @mentioned
        if not self.bot.user or not self.bot.user.mentioned_in(message):
            return

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
                    f"[Polaris] Observed reaction from {observed_agent}: {reaction.emoji}"
                )

    async def _handle_mention(self, message: discord.Message, is_from_agent: bool):
        """Handle when Polaris is @mentioned - queue the message for processing."""
        # Extract message content without the mention
        clean_content = self.agent_messaging.extract_message_for_agent(
            message, "polaris"
        )

        if not clean_content:
            await message.channel.send(
                "You mentioned me but didn't say anything. "
                "How can I help with your calendar?"
            )
            return

        # Log who we received message from
        if is_from_agent:
            sender_agent = self.agent_messaging.is_from_agent(message)
            logger.info(f"Polaris received message from agent: {sender_agent}")

            # Send acknowledgment reaction to let sender know we received it
            # This is visible to all agents and confirms delivery
            if message.id not in self._acknowledged_message_ids:
                try:
                    await message.add_reaction(REACTION_ACK)
                    self._acknowledged_message_ids.add(message.id)
                    logger.info(f"[Polaris] Sent ✅ acknowledgment for message {message.id}")

                    # Keep set from growing indefinitely
                    if len(self._acknowledged_message_ids) > 500:
                        # Remove oldest entries (convert to list, slice, back to set)
                        self._acknowledged_message_ids = set(
                            list(self._acknowledged_message_ids)[-250:]
                        )
                except Exception as e:
                    logger.warning(f"[Polaris] Failed to send acknowledgment: {e}")
        else:
            logger.info(f"Polaris received message from user: {message.author}")

        # Get or create conversation for this channel
        conversation_id = await self._get_or_create_conversation(
            channel_id=message.channel.id,
            user_id=message.author.id,
            guild_id=message.guild.id if message.guild else None,
        )

        # Create agent context
        context = AgentContext(
            channel_id=message.channel.id,
            user_id=message.author.id,
            message_content=clean_content,
            guild_id=message.guild.id if message.guild else None,
            mentioned_agent="polaris",
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
                    f"[Polaris] Duplicate task detected from {self.agent_messaging.is_from_agent(message)}, "
                    f"acknowledged silently (hash match or semantic duplicate)"
                )
            except Exception as e:
                logger.warning(f"[Polaris] Failed to add duplicate ack reaction: {e}")
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
        if not self.agent:
            await message.channel.send("Polaris agent not initialized yet. Please wait.")
            return

        logger.info(
            f"[Polaris] PROCESSING: msg_id={message.id}, from_agent={is_from_agent}, "
            f"content={context.message_content[:50]}..."
        )

        try:
            # Show typing indicator while processing
            async with message.channel.typing():
                response = await self.agent.process(context)

            # Track completed task for workflow continuity
            self.workflow_state_provider.on_task_completed(context.message_content)

            # Send response if there's content
            if response.content:
                logger.info(
                    f"[Polaris] SENDING RESPONSE: msg_id={message.id}, "
                    f"response_len={len(response.content)}, tool_calls={response.tool_calls_made}, "
                    f"content_preview={response.content[:100]}..."
                )
                await self._send_response(message.channel, response.content)
            else:
                logger.warning(
                    f"[Polaris] NO RESPONSE CONTENT: msg_id={message.id}, "
                    f"tool_calls={response.tool_calls_made}"
                )

            # Log processing stats
            logger.info(
                f"[Polaris] PROCESS COMPLETE: msg_id={message.id}, "
                f"time={response.processing_time_ms}ms, tools={response.tool_calls_made}"
            )

            # Send completion reaction for inter-agent messages
            # This tells other agents we've finished processing
            if is_from_agent:
                try:
                    await message.add_reaction(REACTION_DONE)
                    logger.info(f"[Polaris] Sent ✔️ completion for message {message.id}")
                except Exception as e:
                    logger.warning(f"[Polaris] Failed to send completion reaction: {e}")

        except Exception as e:
            logger.error(f"[Polaris] PROCESS ERROR: msg_id={message.id}, error={e}", exc_info=True)
            await message.channel.send(f"Error processing message: {str(e)[:500]}")

    async def _send_response(self, channel: discord.TextChannel, content: str):
        """Send a response, splitting if necessary."""
        max_length = 1900

        if len(content) <= max_length:
            await channel.send(content)
            return

        # Split into chunks
        chunks = []
        while content:
            if len(content) <= max_length:
                chunks.append(content)
                break

            split_at = max_length
            for sep in ["\n", " "]:
                idx = content.rfind(sep, 0, max_length)
                if idx > max_length // 2:
                    split_at = idx + 1
                    break

            chunks.append(content[:split_at])
            content = content[split_at:]

        for chunk in chunks:
            await channel.send(chunk)

    # === Commands ===

    @commands.command(name="cal")
    async def cmd_calendar(self, ctx: commands.Context, *, query: str = ""):
        """Quick calendar command - show today's events or process a query."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        if not query:
            query = "What's on my calendar today?"

        # Create context and process
        context = AgentContext(
            channel_id=ctx.channel.id,
            user_id=ctx.author.id,
            message_content=query,
            guild_id=ctx.guild.id if ctx.guild else None,
            mentioned_agent="polaris",
        )

        async with ctx.typing():
            response = await self.agent.process(context)

        if response.content:
            await self._send_response(ctx.channel, response.content)

    @commands.command(name="pstatus")
    async def cmd_status(self, ctx: commands.Context):
        """Get Polaris's current status."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        from polaris.tools.utils import check_calendar_auth_status

        status_parts = ["**Polaris Status**"]

        # Check calendar auth status
        auth_status = check_calendar_auth_status(
            credentials_path=self.config.google_credentials_path,
            token_path=self.config.google_token_path,
        )

        if self._calendar_service:
            status_parts.append("- Calendar: ✅ Connected")
        elif auth_status['authenticated']:
            status_parts.append("- Calendar: ⚠️ Token valid but service not initialized")
        else:
            status_parts.append(f"- Calendar: ❌ {auth_status['message']}")

        # Token info
        status_parts.append(f"- Token path: `{auth_status['token_path']}`")

        # Config info
        status_parts.append(f"- Timezone: {self.config.timezone}")
        status_parts.append(
            f"- Working hours: {self.config.working_hours_start}:00 - "
            f"{self.config.working_hours_end}:00"
        )

        await ctx.send("\n".join(status_parts))

    @commands.command(name="pauth")
    async def cmd_auth(self, ctx: commands.Context):
        """Trigger Google Calendar OAuth authentication via Discord."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        from polaris.tools.utils import (
            check_calendar_auth_status,
            get_calendar_service,
            start_oauth_flow,
            complete_oauth_flow,
        )

        # Check current status
        auth_status = check_calendar_auth_status(
            credentials_path=self.config.google_credentials_path,
            token_path=self.config.google_token_path,
        )

        if not auth_status['credentials_exists']:
            await ctx.send(
                "❌ **No credentials file found.**\n\n"
                "To set up Google Calendar:\n"
                "1. Go to [Google Cloud Console](https://console.cloud.google.com/)\n"
                "2. Create a project and enable Calendar API\n"
                "3. Create OAuth credentials (Desktop app)\n"
                "4. Download `credentials.json`\n"
                "5. Set `GOOGLE_CREDENTIALS_PATH` to the file path\n"
                "6. Run `!pauth` again"
            )
            return

        if auth_status['authenticated'] and self._calendar_service:
            await ctx.send("✅ Already authenticated! Calendar is connected.")
            return

        # Start OAuth flow and get the authorization URL
        try:
            flow, auth_url = start_oauth_flow(self.config.google_credentials_path)
        except Exception as e:
            await ctx.send(f"❌ Failed to start OAuth: {e}")
            return

        # Store the flow for later completion
        self._pending_oauth_flow = flow

        await ctx.send(
            "🔐 **Google Calendar Authentication**\n\n"
            f"**Click here to authorize:** {auth_url}\n\n"
            "After authorizing, Google will show you a code.\n"
            "Copy that code and run: `!pauth_code <your-code>`"
        )

    @commands.command(name="pauth_code")
    async def cmd_auth_code(self, ctx: commands.Context, *, code: str = None):
        """Complete OAuth by providing the authorization code."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        if not code:
            await ctx.send("❌ Please provide the authorization code: `!pauth_code <code>`")
            return

        if not hasattr(self, '_pending_oauth_flow') or not self._pending_oauth_flow:
            await ctx.send("❌ No pending OAuth flow. Run `!pauth` first to get an authorization link.")
            return

        from polaris.tools.utils import complete_oauth_flow

        try:
            # Complete the OAuth flow with the provided code
            self._calendar_service = complete_oauth_flow(
                flow=self._pending_oauth_flow,
                code=code.strip(),
                token_path=self.config.google_token_path,
            )

            # Clear the pending flow
            self._pending_oauth_flow = None

            # Update agent with new service
            if self.agent:
                self.agent.set_calendar_service(self._calendar_service)

            await ctx.send("✅ **Authentication successful!** Calendar is now connected.")

        except Exception as e:
            logger.error(f"OAuth code exchange failed: {e}", exc_info=True)
            await ctx.send(f"❌ **Authentication failed:** {str(e)}\n\nTry running `!pauth` again.")


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(PolarisCore(bot))
