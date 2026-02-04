"""Canopus Core Cog - Main Discord integration for Canopus bot."""

import asyncio
import logging
from typing import Optional

import discord
from discord.ext import commands, tasks

from shared.base_agent import AgentContext
from shared.agent_messaging import AgentMessaging, parse_node_marker, strip_node_marker
from shared.discord_utils import convert_text_mentions_to_discord
from shared.agent_queue import AgentMessageQueue
from shared.workflow_state import QueueWorkflowStateProvider
from shared.agent_coordinator import (
    DistributedAgentTracker,
    AgentAcknowledgmentMixin,
    REACTION_ACK,
    REACTION_DONE,
)
from canopus.agent import CanopusAgent
from canopus.browser.manager import BrowserSessionManager
from canopus.config import get_config

logger = logging.getLogger(__name__)


class CanopusCore(commands.Cog, AgentAcknowledgmentMixin):
    """
    Core cog for the Canopus Discord bot.

    Handles:
    - Message routing to CanopusAgent when mentioned
    - Inter-agent communication via @mentions
    - Browser session management per channel
    - Cleanup of idle browser sessions
    - Distributed agent tracking and acknowledgment
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
        self.soul_manager = None

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
                from shared.soul import SoulManager, get_default_traits
                import asyncio

                self.memory_manager = MemoryManager(self.config.redis_url)
                self.conversation_manager = ConversationManager(self.config.redis_url)
                self.soul_manager = SoulManager(self.config.redis_url)
                logger.info("Memory, conversation, and soul managers initialized")

                # Initialize default soul traits for Canopus if not already present
                asyncio.create_task(
                    self.soul_manager.ensure_defaults_initialized(
                        "canopus", get_default_traits("canopus")
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Redis-backed managers: {e}")

        # Initialize LLM providers
        from shared.llm.factory import create_llm_provider, create_utility_llm_from_config
        llm = create_llm_provider(
            provider=self.config.llm_provider,
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
        )

        # Create utility LLM for lightweight evaluations
        utility_llm = create_utility_llm_from_config(self.config)

        # Initialize the Canopus agent
        self.agent = CanopusAgent(
            llm=llm,
            browser_manager=self.browser_manager,
            conversation_manager=self.conversation_manager,
            memory_manager=self.memory_manager,
            soul_manager=self.soul_manager,
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

        # Only respond when explicitly @mentioned
        if not self.bot.user or not self.bot.user.mentioned_in(message):
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

        # Pre-enqueue check: Should we respond at all?
        if is_from_agent:
            # Check if this is a genuine task dispatch (mention at start)
            # vs. casual mention in another agent's conversational response.
            # Real dispatches from the job graph executor always start with
            # our @mention. Conversational messages embed mentions mid-text.
            if not self._is_agent_dispatch(message):
                logger.info(
                    f"[Canopus] SKIPPING casual agent mention: '{clean_content[:50]}...' - "
                    f"not a direct dispatch (mention not at start of message)"
                )
                return
        else:
            # Check for casual mentions from users that don't need a response
            should_respond = await self._should_respond_to_mention(message, clean_content)
            if not should_respond:
                logger.info(
                    f"[Canopus] SKIPPING casual mention: '{clean_content[:50]}...' - "
                    f"utility LLM decided no response needed"
                )
                return

        # Extract node marker from dispatch content (if present)
        node_marker = None
        if is_from_agent:
            parsed = parse_node_marker(clean_content)
            if parsed:
                from shared.agent_messaging import format_node_marker
                node_marker = format_node_marker(*parsed)
                clean_content = strip_node_marker(clean_content)
                logger.info(f"[Canopus] Extracted node marker: {node_marker}")

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
            is_from_agent=is_from_agent,
            node_marker=node_marker,
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

        # Record when we started processing for stale check
        processing_started = message.created_at

        # Show typing indicator while processing
        async with message.channel.typing():
            response = await self.agent.process(context)

        # Track completed task for workflow continuity
        self.workflow_state_provider.on_task_completed(context.message_content)

        # Check if response is just noise (acknowledgments, etc.)
        if response.content and self._is_noise_response(response.content):
            logger.info(
                f"[Canopus] SUPPRESSING NOISE RESPONSE: msg_id={message.id}, "
                f"content='{response.content[:50]}'"
            )
            response.content = ""  # Clear the noise

        # Check if our response is stale (conversation moved on while we were processing)
        should_send = True
        if response.content:
            should_send = await self._check_response_freshness(
                message.channel, processing_started, message.id, is_from_agent
            )

        # Send response only if there's content AND response is still relevant
        if response.content and should_send:
            # Prepend @Vega + node marker so Vega can match this to the graph node
            if is_from_agent and context.node_marker:
                vega_id = self.config.agent_registry.agents.get("vega")
                if vega_id:
                    response.content = f"<@{vega_id}> {context.node_marker} {response.content}"

            logger.info(
                f"[Canopus] SENDING RESPONSE: msg_id={message.id}, "
                f"response_len={len(response.content)}, tool_calls={response.tool_calls_made}, "
                f"content_preview={response.content[:100]}..."
            )
            await self._send_response(message.channel, response.content)
        elif response.content and not should_send:
            logger.info(
                f"[Canopus] SUPPRESSING STALE RESPONSE: msg_id={message.id}, "
                f"conversation moved on while processing"
            )
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
        # Only mark done if we actually sent a response
        if is_from_agent and response.content and should_send:
            try:
                await message.add_reaction(REACTION_DONE)
                logger.info(f"[Canopus] Sent ✔️ completion for message {message.id}")
            except Exception as e:
                logger.warning(f"[Canopus] Failed to send completion reaction: {e}")

    async def _check_response_freshness(
        self,
        channel: discord.TextChannel,
        since: any,
        original_msg_id: int,
        is_from_agent: bool,
    ) -> bool:
        """
        Check if our response is still relevant or if the conversation has moved on.

        Returns True if we should send the response, False if it's stale.
        """
        try:
            # Count new messages since we started processing
            new_message_count = 0
            user_messages = 0
            agent_messages = 0

            async for msg in channel.history(limit=15, after=since):
                # Skip our trigger message
                if msg.id == original_msg_id:
                    continue
                # Skip empty messages
                if not msg.content or not msg.content.strip():
                    continue

                new_message_count += 1

                if msg.author.bot:
                    agent_messages += 1
                else:
                    user_messages += 1

            # Decision logic:
            # - If user sent 2+ new messages, conversation likely moved on
            # - If 3+ agent messages came in, conversation is very active
            # - If from an agent (dispatch), we should always respond (it's a task)
            if is_from_agent:
                # Always respond to dispatches, but log if conversation is busy
                if new_message_count > 5:
                    logger.info(
                        f"[Canopus] Conversation busy ({new_message_count} new msgs) "
                        f"but responding to agent dispatch"
                    )
                return True

            if user_messages >= 2:
                logger.info(
                    f"[Canopus] Response stale: {user_messages} user messages since we started"
                )
                return False

            if agent_messages >= 3:
                logger.info(
                    f"[Canopus] Response stale: {agent_messages} agent messages since we started"
                )
                return False

            return True

        except Exception as e:
            logger.warning(f"[Canopus] Freshness check failed: {e}")
            # On error, default to sending the response
            return True

    def _is_agent_dispatch(self, message: discord.Message) -> bool:
        """
        Check if an agent message is a direct task dispatch vs. casual mention.

        Real dispatches from the job graph executor start with the agent's
        @mention followed by a task description:
            "<@123456789> investigate the terminal issue"

        Conversational messages (e.g., Vega's respond_to_user) embed mentions
        within natural language:
            "See? <@123456789> is already getting the hang of the bouncer role"

        Returns True only for direct dispatches.
        """
        if not self.bot.user:
            return True  # Can't check, assume dispatch to be safe

        content = message.content.strip()
        bot_id = str(self.bot.user.id)

        # Dispatch format: message starts with our @mention
        # Covers both <@ID> and <@!ID> (nickname) mention formats
        if content.startswith(f"<@{bot_id}>") or content.startswith(f"<@!{bot_id}>"):
            return True

        return False

    async def _should_respond_to_mention(
        self,
        message: discord.Message,
        clean_content: str,
    ) -> bool:
        """
        Use utility LLM to quickly decide if we should respond to this mention.

        Returns False for casual mentions, greetings, or when we're just being
        discussed rather than asked to do something.
        """
        # Quick pattern checks for obvious cases (save LLM call)
        content_lower = clean_content.lower().strip()

        # Always respond to web/browse-related queries
        if any(word in content_lower for word in [
            "browse", "search", "look up", "find", "website", "page", "url",
            "click", "navigate", "go to", "open", "screenshot", "extract",
            "scrape", "read", "research", "web", "google", "bing",
        ]):
            return True

        # Check if utility LLM is available
        if not self.agent or not self.agent.utility_llm:
            # No utility LLM, default to responding
            return True

        # Gather recent context
        recent_context = []
        try:
            async for msg in message.channel.history(limit=5, before=message):
                if msg.content:
                    author = msg.author.display_name
                    recent_context.append(f"[{author}]: {msg.content[:100]}")
            recent_context.reverse()
        except Exception:
            pass

        context_str = "\n".join(recent_context[-3:]) if recent_context else "(no recent messages)"

        # Quick LLM evaluation
        from shared.llm.types import Message, Role
        prompt = f"""You are deciding whether an AI agent named Canopus should respond to a Discord message.

Canopus is a web browser/research specialist. He should ONLY respond when:
- Someone is asking him to browse, search, or research something online
- Someone is asking him to interact with a webpage (click, fill forms, screenshot)
- He is being assigned a web-related task

Canopus should NOT respond when:
- People are just chatting casually and happened to mention him
- It's a general greeting to everyone (like "hi everyone" or "hello team")
- Others are discussing him but not talking TO him
- Another agent already handled the request
- It's just social pleasantries or acknowledgments

Recent conversation:
{context_str}

Message that mentioned Canopus:
[{message.author.display_name}]: {clean_content}

Should Canopus respond to this message? Answer only YES or NO."""

        try:
            response = await self.agent.utility_llm.generate(
                messages=[Message(role=Role.USER, content=prompt)],
                temperature=0.0,
                max_tokens=10,
            )

            answer = response.content.strip().upper() if response.content else "YES"
            should_respond = "YES" in answer

            if not should_respond:
                logger.info(
                    f"[Canopus] Utility LLM says NO RESPONSE needed for: "
                    f"'{clean_content[:40]}...'"
                )

            return should_respond

        except Exception as e:
            logger.warning(f"[Canopus] Utility LLM check failed: {e}")
            # On error, default to responding
            return True

    def _is_noise_response(self, content: str) -> bool:
        """
        Check if a response is just noise (acknowledgments, short phrases).
        Returns True if the response should be suppressed.
        """
        if not content:
            return False

        # Normalize
        normalized = content.strip().lower()

        # Remove punctuation for comparison
        import re
        clean = re.sub(r'[^\w\s]', '', normalized)

        # Exact match noise phrases
        noise_phrases = {
            "done", "task completed", "acknowledged", "understood",
            "copy that", "ready", "got it", "noted", "roger",
            "affirmative", "on it", "will do", "ok", "okay",
            "yes", "yep", "yup", "sure", "alright", "confirmed",
            "received", "standing by", "waiting", "listening",
            "bouncer protocol integrated", "bouncer protocol active",
            "protocol acknowledged", "protocol integrated",
            "web operation completed", "operation completed",
            "task done", "completed", "finished", "task finished",
        }

        if clean in noise_phrases:
            return True

        # Check if response is very short (likely noise)
        # But only if it doesn't contain meaningful content
        if len(clean.split()) <= 3:
            # Check for common noise patterns
            noise_patterns = [
                r"^(done|ok|okay|yes|no|sure|ready|got it|noted)\.?$",
                r"^task (completed|done|finished)\.?$",
                r"^(acknowledged|understood|copy that|roger)\.?$",
                r"^(web )?operation (completed|done)\.?$",
                r"^(standing by|waiting|ready)\.?$",
                r"^(affirmative|confirmed|received)\.?$",
                r"^protocol (integrated|acknowledged|active)\.?$",
                r"^bouncer protocol.*$",
            ]
            for pattern in noise_patterns:
                if re.match(pattern, normalized):
                    return True

        return False

    async def _send_response(self, channel: discord.TextChannel, content: str):
        """Send a response, splitting if necessary."""
        max_length = 1900
        max_total_length = 6000  # Absolute max to prevent spam

        # Convert @Name text to actual Discord mentions
        content = await convert_text_mentions_to_discord(
            content,
            channel,
            self.config.agent_registry.agents,
        )

        # Detect and truncate repetitive content
        content = self._clean_repetitive_content(content)

        # Enforce total length limit
        if len(content) > max_total_length:
            content = content[:max_total_length] + "\n\n... [Response truncated due to length]"

        if len(content) <= max_length:
            await channel.send(content, silent=True)
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
            await channel.send(chunk, silent=True)

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
