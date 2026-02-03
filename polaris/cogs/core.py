"""Polaris Core Cog - Main Discord integration for Polaris bot."""

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
        self.soul_manager = None
        self.todo_manager = None
        self.reminder_manager = None
        self.calendar_sync_manager = None
        self.adaptation_engine = None
        self.notification_manager = None

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

        # Start reminder checker background task
        if self.reminder_manager:
            self.reminder_checker.start()
            logger.info("Reminder checker background task started")

        logger.info("Polaris Core cog initialized successfully")

    def cog_unload(self):
        """Clean up when cog is unloaded."""
        self.message_queue.stop()
        if self.reminder_checker.is_running():
            self.reminder_checker.cancel()

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
                from shared.soul import SoulManager, get_default_traits
                from polaris.todo import TodoManager
                from polaris.todo.reminders import ReminderManager
                import asyncio

                self.memory_manager = MemoryManager(self.config.redis_url)
                self.conversation_manager = ConversationManager(self.config.redis_url)
                self.soul_manager = SoulManager(self.config.redis_url)
                self.todo_manager = TodoManager(self.config.redis_url)
                self.reminder_manager = ReminderManager(self.config.redis_url)

                from polaris.todo.adaptation import AdaptationEngine
                self.adaptation_engine = AdaptationEngine(self.config.redis_url)
                logger.info("Memory, conversation, soul, todo, reminder, and adaptation managers initialized")

                # Initialize default soul traits for Polaris if not already present
                asyncio.create_task(
                    self.soul_manager.ensure_defaults_initialized(
                        "polaris", get_default_traits("polaris")
                    )
                )
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
            soul_manager=self.soul_manager,
            discord_bot=self.bot,
            agent_registry=self.config.agent_registry.agents,
            config=self.config,
            utility_llm=utility_llm,
        )

        # Set calendar service if available
        if self._calendar_service:
            self.agent.set_calendar_service(self._calendar_service)

        # Set todo manager and calendar sync if available
        if self.todo_manager:
            self.agent.todo_manager = self.todo_manager

            # Initialize calendar sync manager
            if self._calendar_service:
                from polaris.todo.calendar_sync import CalendarSyncManager
                self.calendar_sync_manager = CalendarSyncManager(
                    redis_url=self.config.redis_url,
                    calendar_service=self._calendar_service,
                    config=self.config,
                )
                self.agent.calendar_sync_manager = self.calendar_sync_manager
                logger.info("Todo manager + calendar sync connected to Polaris agent")
            else:
                logger.info("Todo manager connected (calendar sync unavailable - no calendar service)")

        # Set reminder manager if available
        if self.reminder_manager:
            self.agent.reminder_manager = self.reminder_manager
            logger.info("Reminder manager connected to Polaris agent")

        # Set adaptation engine if available
        if self.adaptation_engine:
            self.agent.adaptation_engine = self.adaptation_engine
            # Wire completion tracker into todo manager for automatic tracking
            if self.todo_manager:
                self.todo_manager._completion_tracker = self.adaptation_engine.tracker
            logger.info("Adaptation engine connected to Polaris agent")

        # Initialize notification manager (Discord DM + Web Push)
        push_service = None
        if self.config.redis_url:
            try:
                from polaris.notifications.push import PushNotificationService
                from polaris.api.config import get_api_config

                api_config = get_api_config()
                if api_config.vapid_private_key:
                    push_service = PushNotificationService(
                        redis_url=self.config.redis_url,
                        vapid_private_key=api_config.vapid_private_key,
                        vapid_public_key=api_config.vapid_public_key,
                        vapid_subject=api_config.vapid_subject,
                    )
                    logger.info("Web Push notification service initialized")
            except Exception as e:
                logger.debug(f"Web Push not available: {e}")

        from polaris.notifications.manager import NotificationManager
        self.notification_manager = NotificationManager(
            bot=self.bot,
            push_service=push_service,
            fallback_user_id=self.config.allowed_user_id,
            default_channel_id=self.config.main_channel_id,
        )
        self.agent.notification_manager = self.notification_manager
        logger.info("Notification manager connected to Polaris agent")

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

        # Pre-enqueue check: Should we respond at all?
        if is_from_agent:
            # Check if this is a genuine task dispatch (mention at start)
            # vs. casual mention in another agent's conversational response.
            # Real dispatches from the job graph executor always start with
            # our @mention. Conversational messages embed mentions mid-text.
            if not self._is_agent_dispatch(message):
                logger.info(
                    f"[Polaris] SKIPPING casual agent mention: '{clean_content[:50]}...' - "
                    f"not a direct dispatch (mention not at start of message)"
                )
                return
        else:
            # Check for casual mentions from users that don't need a response
            should_respond = await self._should_respond_to_mention(message, clean_content)
            if not should_respond:
                logger.info(
                    f"[Polaris] SKIPPING casual mention: '{clean_content[:50]}...' - "
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
                logger.info(f"[Polaris] Extracted node marker: {node_marker}")

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

        # Record when we started processing for stale check
        processing_started = message.created_at

        try:
            # Show typing indicator while processing
            async with message.channel.typing():
                response = await self.agent.process(context)

            # Track completed task for workflow continuity
            self.workflow_state_provider.on_task_completed(context.message_content)

            # Check if response is just noise (acknowledgments, etc.)
            if response.content and self._is_noise_response(response.content):
                logger.info(
                    f"[Polaris] SUPPRESSING NOISE RESPONSE: msg_id={message.id}, "
                    f"content='{response.content[:50]}'"
                )
                response.content = ""  # Clear the noise

            # Check if our response is stale (conversation moved on while we were processing)
            should_send = True
            if response.content:
                should_send = await self._check_response_freshness(
                    message.channel, processing_started, message.id, is_from_agent
                )

            # Send response if there's content AND response is still relevant
            if response.content and should_send:
                # Prepend @Vega + node marker so Vega can match this to the graph node
                if is_from_agent and context.node_marker:
                    vega_id = self.config.agent_registry.agents.get("vega")
                    if vega_id:
                        response.content = f"<@{vega_id}> {context.node_marker} {response.content}"

                logger.info(
                    f"[Polaris] SENDING RESPONSE: msg_id={message.id}, "
                    f"response_len={len(response.content)}, tool_calls={response.tool_calls_made}, "
                    f"content_preview={response.content[:100]}..."
                )
                await self._send_response(message.channel, response.content)
            elif response.content and not should_send:
                logger.info(
                    f"[Polaris] SUPPRESSING STALE RESPONSE: msg_id={message.id}, "
                    f"conversation moved on while processing"
                )
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
                        f"[Polaris] Conversation busy ({new_message_count} new msgs) "
                        f"but responding to agent dispatch"
                    )
                return True

            if user_messages >= 2:
                logger.info(
                    f"[Polaris] Response stale: {user_messages} user messages since we started"
                )
                return False

            if agent_messages >= 3:
                logger.info(
                    f"[Polaris] Response stale: {agent_messages} agent messages since we started"
                )
                return False

            return True

        except Exception as e:
            logger.warning(f"[Polaris] Freshness check failed: {e}")
            # On error, default to sending the response
            return True

    def _is_agent_dispatch(self, message: discord.Message) -> bool:
        """
        Check if an agent message is a direct task dispatch vs. casual mention.

        Real dispatches from the job graph executor start with the agent's
        @mention followed by a task description:
            "<@123456789> check the calendar for tomorrow"

        Conversational messages (e.g., Vega's respond_to_user) embed mentions
        within natural language:
            "I'll have <@123456789> check your schedule later"

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

        # Always respond to calendar or todo-related queries
        if any(word in content_lower for word in [
            "calendar", "schedule", "event", "meeting", "appointment",
            "when", "what time", "free", "busy", "book", "cancel",
            "reschedule", "remind", "today", "tomorrow", "week",
            "todo", "to-do", "task", "due", "complete", "done",
            "reminder", "overdue", "priority",
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
        prompt = f"""You are deciding whether an AI agent named Polaris should respond to a Discord message.

Polaris is a calendar/scheduling specialist. She should ONLY respond when:
- Someone is asking about calendar, scheduling, meetings, or events
- Someone is asking her a direct question
- She is being assigned a task related to time management

Polaris should NOT respond when:
- People are just chatting casually and happened to mention her
- It's a general greeting to everyone (like "hi everyone" or "hello team")
- Others are discussing her but not talking TO her
- Another agent already handled the request
- It's just social pleasantries or acknowledgments

Recent conversation:
{context_str}

Message that mentioned Polaris:
[{message.author.display_name}]: {clean_content}

Should Polaris respond to this message? Answer only YES or NO."""

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
                    f"[Polaris] Utility LLM says NO RESPONSE needed for: "
                    f"'{clean_content[:40]}...'"
                )

            return should_respond

        except Exception as e:
            logger.warning(f"[Polaris] Utility LLM check failed: {e}")
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
        """Send a response, splitting if necessary. Sent silently (no push notification)."""
        max_length = 1900

        # Convert @Name text to actual Discord mentions
        content = await convert_text_mentions_to_discord(
            content,
            channel,
            self.config.agent_registry.agents,
        )

        if len(content) <= max_length:
            await channel.send(content, silent=True)
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
            await channel.send(chunk, silent=True)

    # === Background Tasks ===

    @tasks.loop(seconds=30)
    async def reminder_checker(self):
        """Check for due reminders and send notifications."""
        if not self.reminder_manager:
            return

        try:
            due_reminders = await self.reminder_manager.get_due_reminders()
            if not due_reminders:
                return

            for reminder in due_reminders:
                try:
                    await self._fire_reminder(reminder)
                    await self.reminder_manager.mark_fired(reminder.id)
                except Exception as e:
                    logger.error(f"Failed to fire reminder {reminder.id}: {e}")

        except Exception as e:
            logger.error(f"Reminder checker error: {e}", exc_info=True)

    @reminder_checker.before_loop
    async def before_reminder_checker(self):
        """Wait for bot to be ready before starting the loop."""
        await self.bot.wait_until_ready()

    async def _fire_reminder(self, reminder):
        """Send a reminder notification via NotificationManager."""
        # Build extra context from linked todo
        extra_text = ""
        todo_context = ""
        if reminder.todo_id and self.todo_manager:
            todo = await self.todo_manager.get(reminder.todo_id)
            if todo:
                parts = [f"Todo: **{todo.title}** [{todo.id}]"]
                if todo.due_at:
                    parts.append(f"Due: {todo.due_at.strftime('%b %d %I:%M %p')}")
                extra_text = "\n".join(parts)
                todo_context = f"Linked todo: '{todo.title}'"
                if todo.due_at:
                    todo_context += f" (due {todo.due_at.strftime('%b %d %I:%M %p')})"

        # Generate natural reminder text via utility LLM
        formatted_text = ""
        if self.agent and self.agent.utility_llm:
            try:
                from shared.llm.types import Message, Role

                snooze_info = ""
                if getattr(reminder, 'snooze_count', 0) > 0:
                    snooze_info = f"\nThis reminder has been snoozed {reminder.snooze_count} time(s)."

                prompt = (
                    f"Write a short, friendly Discord reminder message (1-2 sentences). "
                    f"Do NOT use markdown bold on the word 'Reminder'. "
                    f"Be natural and conversational.\n\n"
                    f"Reminder: {reminder.message}\n"
                    f"{todo_context}\n"
                    f"{snooze_info}\n"
                    f"Write ONLY the reminder text, nothing else."
                )

                response = await self.agent.utility_llm.generate(
                    messages=[Message(role=Role.USER, content=prompt)],
                    temperature=0.7,
                    max_tokens=150,
                )
                if response.content and response.content.strip():
                    formatted_text = response.content.strip()
                    logger.debug(f"LLM-generated reminder text for {reminder.id}: {formatted_text}")
            except Exception as e:
                logger.warning(f"LLM reminder generation failed for {reminder.id}: {e}")

        if self.notification_manager:
            results = await self.notification_manager.send_reminder(
                reminder, extra_text=extra_text, formatted_text=formatted_text,
            )
            logger.info(f"Fired reminder {reminder.id}: {results}")
        else:
            # Fallback: direct DM if notification manager not available
            user_id = int(reminder.user_id) if reminder.user_id.isdigit() else None
            if user_id:
                try:
                    user = await self.bot.fetch_user(user_id)
                    if user:
                        text = formatted_text or f"**Reminder:** {reminder.message}"
                        if extra_text:
                            text = f"{text}\n{extra_text}"
                        await user.send(text)
                except Exception as e:
                    logger.error(f"Failed to send fallback DM for reminder {reminder.id}: {e}")

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
