"""Altair Core Cog - Main Discord integration for Altair bot."""

import asyncio
import logging
import time
from typing import Optional

import discord
from discord.ext import commands, tasks

from shared.base_agent import AgentContext
from shared.agent_messaging import AgentMessaging
from shared.agent_queue import AgentMessageQueue
from shared.channel import ChannelManager
from shared.database import InMemorySessionStore, FileSessionStore
from shared.session import SessionRegistry, SessionDefaults
from shared.utils import VirtualScreen
from shared.terminal.output_manager import LiveTerminalDisplay
from shared.workflow_state import QueueWorkflowStateProvider
from shared.agent_coordinator import (
    DistributedAgentTracker,
    AgentAcknowledgmentMixin,
    REACTION_ACK,
    REACTION_DONE,
)
from shared.collaboration import CollaborativeCogMixin
from altair.agent import AltairAgent
from altair.config import get_config
from altair.permission import PermissionManager

logger = logging.getLogger(__name__)


class AltairCore(commands.Cog, CollaborativeCogMixin, AgentAcknowledgmentMixin):
    """
    Core cog for the Altair Discord bot.

    Handles:
    - Message routing to AltairAgent when mentioned
    - Inter-agent communication via @mentions
    - Permission requests for sensitive actions
    - Session management integration
    - VLI terminal commands
    - Distributed agent tracking and acknowledgment
    - Collaborative chime-in evaluation for proactive agent participation
    """

    # VLI command constants
    SPECIAL_KEYS = {
        'up', 'down', 'left', 'right',
        'enter', 'esc', 'escape',
        'backspace', 'tab',
        'c', 'd', 'z', 'l'
    }

    CONTROL_ALIASES = {
        'ctrl-c': 'c',
        'ctrl-d': 'd',
        'ctrl-z': 'z',
        'ctrl-l': 'l',
    }

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = get_config()

        # Components initialized in cog_load
        self.agent: Optional[AltairAgent] = None
        self.permission_manager: Optional[PermissionManager] = None
        self.agent_messaging: Optional[AgentMessaging] = None

        # Session and channel management
        self._store = None
        self.session_registry = None
        self.channel_manager = ChannelManager(bot)
        self.project_manager = None
        self.memory_manager = None
        self.conversation_manager = None

        # Track per-session state for output loops
        self._output_tasks: dict = {}

        # Workflow state provider for context awareness
        self.workflow_state_provider = QueueWorkflowStateProvider("Altair")

        # Message queue for sequential processing
        self.message_queue = AgentMessageQueue(
            agent_name="Altair",
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
        logger.info("Initializing Altair Core cog...")

        # Initialize components
        await self._init_components()

        # Initialize distributed tracker for inter-agent communication
        self.tracker = DistributedAgentTracker(
            own_name="altair",
            agent_registry=self.config.agent_registry.agents,
        )
        logger.info("Distributed agent tracker initialized")

        # Recover any existing sessions from persistent store
        await self._recover_sessions()

        # Start the message queue processor
        self.message_queue.start()

        logger.info("Altair Core cog initialized successfully")

    async def _recover_sessions(self):
        """Recover sessions from persistent store on startup."""
        try:
            recovered = await self.session_registry.recover_sessions(
                start_output_loop_callback=self.start_output_loop
            )
            if recovered:
                logger.info(f"Recovered {len(recovered)} sessions from persistent store")
        except Exception as e:
            logger.error(f"Failed to recover sessions: {e}")

    def cog_unload(self):
        """Clean up when cog is unloaded."""
        self.health_check.cancel()
        self.message_queue.stop()
        asyncio.create_task(self.session_registry.shutdown_all())

    async def _init_components(self):
        """Initialize all components."""
        # Initialize permission manager
        self.permission_manager = PermissionManager(
            bot=self.bot,
            allowed_user_id=self.config.allowed_user_id,
            timeout=self.config.permission_timeout,
        )

        # Initialize agent messaging
        self.agent_messaging = AgentMessaging(
            bot=self.bot,
            agent_registry=self.config.agent_registry.agents,
            own_agent_name="altair",
        )

        # Initialize session store and registry
        self._store = self._create_session_store()
        session_defaults = SessionDefaults(
            cli_command=self.config.cli_command,
            workspace_dir=self.config.workspace_dir,
            terminal_cols=self.config.terminal_cols,
            terminal_rows=self.config.terminal_rows,
            terminal_backend=self.config.terminal_backend,
        )
        self.session_registry = SessionRegistry(self._store, session_defaults)

        # Set death callback for session cleanup
        self.session_registry.set_death_callback(self._on_process_death)

        # Start health check loop
        self.health_check.start()

        logger.info("Session registry initialized")

        # Initialize Redis-backed managers if available
        if self.config.redis_url:
            try:
                from shared.project import ProjectManager
                from shared.memory import MemoryManager
                from shared.conversation import ConversationManager
                self.project_manager = ProjectManager(self.config.redis_url)
                self.memory_manager = MemoryManager(self.config.redis_url)
                self.conversation_manager = ConversationManager(self.config.redis_url)
                logger.info("Project, memory, and conversation managers initialized")
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

        # Initialize the Altair agent
        self.agent = AltairAgent(
            llm=llm,
            session_registry=self.session_registry,
            project_manager=self.project_manager,
            memory_manager=self.memory_manager,
            discord_bot=self.bot,
            channel_manager=self.channel_manager,
            agent_registry=self.config.agent_registry.agents,
            utility_llm=utility_llm,
        )

        # Connect permission manager, output loop callback, and workflow state to agent
        self.agent.set_permission_manager(self.permission_manager)
        self.agent.set_output_loop_callback(self.start_output_loop)
        self.agent.set_workflow_state_provider(self.workflow_state_provider)

    def _create_session_store(self):
        """Create session store based on config.

        Uses FileSessionStore by default for persistence across restarts.
        Falls back to InMemorySessionStore only if file store fails.
        """
        if self.config.redis_url:
            try:
                from shared.database import RedisSessionStore
                logger.info(f"Using Redis session store: {self.config.redis_url}")
                return RedisSessionStore(self.config.redis_url)
            except ImportError as e:
                logger.warning(f"Redis not available: {e}")

        # Use file-based storage for persistence
        try:
            store = FileSessionStore()
            logger.info(f"Using file session store for persistence")
            return store
        except Exception as e:
            logger.warning(f"File session store failed, using in-memory: {e}")
            return InMemorySessionStore()

    @tasks.loop(seconds=5.0)
    async def health_check(self):
        """Periodic health check for dead processes."""
        await self.session_registry.check_health()

    @health_check.before_loop
    async def before_health_check(self):
        await self.bot.wait_until_ready()

    async def _on_process_death(self, session):
        """Callback when a process dies unexpectedly."""
        logger.info(f"Process death detected for session {session.session_id}")

        channel = self.bot.get_channel(session.channel_id)
        if channel:
            try:
                await channel.send("**Process exited.** Channel will be deleted in 10 seconds.")
                await asyncio.sleep(10)
            except discord.HTTPException:
                pass

        await self._cleanup_session(session.session_id)

    async def _cleanup_session(self, session_id: int):
        """Full cleanup: terminate session and delete channel."""
        session = await self.session_registry.get_session(session_id)
        if session:
            channel_id = session.channel_id

            if session_id in self._output_tasks:
                self._output_tasks[session_id].cancel()
                try:
                    await self._output_tasks[session_id]
                except asyncio.CancelledError:
                    pass
                del self._output_tasks[session_id]

            await self.session_registry.terminate_session(session_id)
            await self.channel_manager.delete_channel(channel_id)

    def start_output_loop(self, session):
        """Start the output loop for a session."""
        task = self.bot.loop.create_task(self._session_output_loop(session))
        self._output_tasks[session.session_id] = task

    async def _session_output_loop(self, session):
        """
        Background loop to send terminal output to Discord channel.

        Uses LiveTerminalDisplay for rate-limited single-message updates:
        - Single message that updates in place (like a real terminal)
        - Rate-limited to respect Discord's 50 edits/minute limit
        - Smart truncation when content exceeds Discord's 2000 char limit
        """
        await self.bot.wait_until_ready()

        channel = self.bot.get_channel(session.channel_id)
        if not channel:
            return

        # Create live display for this session
        display = LiveTerminalDisplay(channel, session.session_id)

        while session.is_alive and display.is_running:
            try:
                # Try to get output from queue with timeout
                try:
                    raw_output = await asyncio.wait_for(
                        session.terminal.output_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # No new output - flush any pending updates
                    await display.flush_pending()
                    continue

                # Update the live display with cleaned content
                cleaned = VirtualScreen.clean_ansi_colors(raw_output)
                await display.update(cleaned)

                # Also update the VirtualScreen for backward compatibility
                session.screen = VirtualScreen()
                session.screen.write(cleaned)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Output loop error for session {session.session_id}: {e}")
                await asyncio.sleep(0.5)

        # Cleanup: flush any remaining pending content
        await display.cleanup()

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
                logger.debug(f"[Altair] Observed activity from {observed_agent}")

        # Check if from allowed user or another agent
        is_allowed_user = message.author.id == self.config.allowed_user_id
        is_from_agent = self.agent_messaging.is_from_agent(message) is not None

        if not is_allowed_user and not is_from_agent:
            return

        content = message.content.strip()

        # Handle !vp commands (project management)
        if content.startswith('!vp'):
            await self._handle_vp_command(message, content)
            return

        # Handle !vli commands
        if content.lower().startswith('!vli'):
            await self._handle_vli_command(message, content)
            return

        # Handle plain text in CLI session channels (send directly to terminal)
        session = await self.session_registry.get_session_by_channel(message.channel.id)
        if session and session.is_alive and not content.startswith('!'):
            await session.terminal.send_input(content + '\n')
            return

        # Check if this bot is mentioned
        if not self.bot.user or not self.bot.user.mentioned_in(message):
            # Not mentioned - evaluate chime-in for messages in main channel
            if self.main_channel_id and message.channel.id == self.main_channel_id:
                if self.agent:
                    logger.debug(f"[Altair] Checking chime-in for: {message.content[:50]}...")
                    await self.evaluate_and_maybe_chime_in(message)
                else:
                    logger.warning(f"[Altair] Agent not initialized, skipping chime-in")
            return

        # Process the mention
        await self._handle_mention(message, is_from_agent)

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

    async def _handle_mention(self, message: discord.Message, is_from_agent: bool):
        """Handle when Altair is @mentioned - queue the message for processing."""
        # Extract message content without the mention
        clean_content = self.agent_messaging.extract_message_for_agent(
            message, "altair"
        )

        if not clean_content:
            await message.channel.send(
                "You mentioned me but didn't say anything. How can I help?"
            )
            return

        # Log who we received message from
        if is_from_agent:
            sender_agent = self.agent_messaging.is_from_agent(message)
            logger.info(f"Altair received message from agent: {sender_agent}")

            # Send acknowledgment reaction to let sender know we received it
            # This is visible to all agents and confirms delivery
            if message.id not in self._acknowledged_message_ids:
                try:
                    await message.add_reaction(REACTION_ACK)
                    self._acknowledged_message_ids.add(message.id)
                    logger.info(f"[Altair] Sent ✅ acknowledgment for message {message.id}")

                    # Keep set from growing indefinitely
                    if len(self._acknowledged_message_ids) > 500:
                        # Remove oldest entries
                        self._acknowledged_message_ids = set(
                            list(self._acknowledged_message_ids)[-250:]
                        )
                except Exception as e:
                    logger.warning(f"[Altair] Failed to send acknowledgment: {e}")
        else:
            logger.info(f"Altair received message from user: {message.author}")

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
            mentioned_agent="altair",
            conversation_id=conversation_id,
        )

        # Add to queue for sequential processing
        # No hardcoded acknowledgments - the LLM will see new messages via
        # _fetch_new_messages_since() and decide how to handle them
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
                    f"[Altair] Duplicate task detected from {self.agent_messaging.is_from_agent(message)}, "
                    f"acknowledged silently (hash match or semantic duplicate)"
                )
            except Exception as e:
                logger.warning(f"[Altair] Failed to add duplicate ack reaction: {e}")
            return

        if not success and status not in ("duplicate_id", "duplicate_task"):
            # Only notify on actual failures (queue full), not duplicates
            await message.channel.send("I'm overwhelmed right now. Please try again shortly.")

    async def _process_queued_message(
        self,
        message: discord.Message,
        context: AgentContext,
        is_from_agent: bool,
    ):
        """Process a message from the queue - this is the actual work."""
        logger.info(
            f"[Altair] PROCESSING: msg_id={message.id}, from_agent={is_from_agent}, "
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
                f"[Altair] SENDING RESPONSE: msg_id={message.id}, "
                f"response_len={len(response.content)}, tool_calls={response.tool_calls_made}, "
                f"content_preview={response.content[:100]}..."
            )
            await self._send_response(message.channel, response.content)
        else:
            logger.info(
                f"[Altair] NO RESPONSE CONTENT: msg_id={message.id}, "
                f"tool_calls={response.tool_calls_made}"
            )

        # Log processing stats
        logger.info(
            f"[Altair] PROCESS COMPLETE: msg_id={message.id}, "
            f"time={response.processing_time_ms}ms, tools={response.tool_calls_made}"
        )

        # Send completion reaction for inter-agent messages
        # This tells other agents we've finished processing
        if is_from_agent:
            try:
                await message.add_reaction(REACTION_DONE)
                logger.info(f"[Altair] Sent ✔️ completion for message {message.id}")
            except Exception as e:
                logger.warning(f"[Altair] Failed to send completion reaction: {e}")

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

            # Find a good split point
            split_at = max_length
            for sep in ['\n', ' ']:
                idx = content.rfind(sep, 0, max_length)
                if idx > max_length // 2:
                    split_at = idx + 1
                    break

            chunks.append(content[:split_at])
            content = content[split_at:]

        for chunk in chunks:
            await channel.send(chunk)

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        """Handle reaction additions for permission system and agent tracking."""
        # Ignore bot's own reactions
        if user == self.bot.user:
            return

        # Track reactions from other agents for acknowledgment detection
        if self.tracker:
            observed_agent = self.tracker.observe_reaction(
                message_id=reaction.message.id,
                user_id=user.id,
                emoji=str(reaction.emoji)
            )
            if observed_agent:
                logger.debug(
                    f"[Altair] Observed reaction from {observed_agent}: {reaction.emoji}"
                )

        # Permission system: only process reactions from allowed user
        if user.id != self.config.allowed_user_id:
            return

        # Let permission manager handle it
        if self.permission_manager:
            self.permission_manager.handle_reaction(reaction, user)

    # === Commands ===

    @commands.command(name="status")
    async def cmd_status(self, ctx: commands.Context):
        """Get Altair's current status."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        if self.agent:
            status = await self.agent.get_status_summary()
            await ctx.send(f"**Altair Status**\n{status}")
        else:
            await ctx.send("Altair agent not initialized.")

    @commands.command(name="sessions")
    async def cmd_sessions(self, ctx: commands.Context):
        """List active terminal sessions."""
        if ctx.author.id != self.config.allowed_user_id:
            return

        if not self.session_registry:
            await ctx.send("Session registry not available.")
            return

        try:
            sessions = await self.session_registry.list_sessions()
            if not sessions:
                await ctx.send("No active sessions.")
                return

            lines = ["**Active Sessions:**"]
            for s in sessions:
                status = "running" if s.is_alive else "stopped"
                channel_ref = f"<#{s.channel_id}>" if s.channel_id else "(no channel)"
                project_info = f" [{s.data.project_name}]" if s.data.project_name else ""
                lines.append(f"- #{s.session_id}: [{status}]{project_info} {channel_ref} PID {s.data.pid}")

            await ctx.send("\n".join(lines))
        except Exception as e:
            await ctx.send(f"Error listing sessions: {e}")

    # --------------------------------------------------
    # VLI COMMANDS
    # --------------------------------------------------

    async def _handle_vli_command(self, message, content: str):
        """Route !vli commands."""
        parts = content.split(None, 2)

        if len(parts) == 1:
            await self._cmd_new_session(message)
            return

        subcommand = parts[1].strip()
        subcommand_lower = subcommand.lower()
        extra = parts[2] if len(parts) > 2 else ""

        if subcommand_lower == 'ls':
            await self._cmd_list_sessions(message)
        elif subcommand_lower == 'quit':
            await self._cmd_send_key(message, 'c')
        elif subcommand_lower == 'exit':
            await self._cmd_exit_session(message, extra)
        elif subcommand_lower in self.CONTROL_ALIASES:
            await self._cmd_send_key(message, self.CONTROL_ALIASES[subcommand_lower])
        elif subcommand_lower == 'status':
            await self._cmd_status(message)
        elif subcommand_lower == 'attach':
            await self._cmd_attach(message)
        elif subcommand_lower == 'resize':
            await self._cmd_resize(message, extra)
        elif subcommand_lower == 'scroll':
            await self._cmd_scroll(message, extra)
        elif subcommand_lower == 'clear':
            await self._cmd_clear(message)
        elif subcommand_lower in self.SPECIAL_KEYS:
            await self._cmd_send_key(message, subcommand_lower)
        elif self.project_manager:
            project = await self.project_manager.get(subcommand)
            if project:
                await self._cmd_new_session_with_project(message, project)
            else:
                await message.channel.send(f"Unknown command or project: `{subcommand}`")
        else:
            await message.channel.send(f"Unknown command: `{subcommand}`")

    async def _cmd_new_session(self, message, workspace_dir=None, command=None, project_name=None):
        """Create a new terminal session."""
        guild = message.guild
        if not guild:
            await message.channel.send("This command only works in a server.")
            return

        next_id = await self.session_registry.get_next_id()
        channel = await self.channel_manager.create_channel(guild, next_id)
        if not channel:
            await message.channel.send("Failed to create channel.")
            return

        try:
            session = await self.session_registry.create_session(
                channel_id=channel.id,
                command=command or self.config.cli_command,
                workspace_dir=workspace_dir or self.config.workspace_dir,
                cols=self.config.terminal_cols,
                rows=self.config.terminal_rows,
                project_name=project_name
            )
        except Exception as e:
            await self.channel_manager.delete_channel(channel.id)
            await message.channel.send(f"Failed to start session: {e}")
            return

        self.start_output_loop(session)

        attach_cmd = session.get_attach_command()
        attach_info = f"\n**Local attach:** `{attach_cmd}`" if attach_cmd else ""
        project_info = f"\n**Project:** {project_name}" if project_name else ""

        await message.channel.send(
            f"**Session #{session.session_id} started.**\n"
            f"Go to {channel.mention} to interact.{project_info}{attach_info}"
        )

    async def _cmd_new_session_with_project(self, message, project):
        """Create a new session for a specific project."""
        command = project.get_setting('default_command') if hasattr(project, 'get_setting') else None
        await self._cmd_new_session(
            message,
            workspace_dir=project.path,
            command=command,
            project_name=project.name
        )

    async def _cmd_list_sessions(self, message):
        """List all active sessions."""
        sessions = await self.session_registry.list_sessions()
        if not sessions:
            await message.channel.send("No active sessions.")
            return

        embed = discord.Embed(title="Active Sessions", color=0x00ff00)
        for session in sessions:
            channel = self.bot.get_channel(session.channel_id)
            channel_ref = channel.mention if channel else "(deleted)"
            status = "running" if session.is_alive else "stopped"
            project_info = f" [{session.data.project_name}]" if session.data.project_name else ""
            embed.add_field(
                name=f"#{session.session_id} [{status}]{project_info}",
                value=f"Channel: {channel_ref}\nPID: {session.data.pid}",
                inline=False
            )
        await message.channel.send(embed=embed)

    async def _cmd_exit_session(self, message, extra: str):
        """Exit/terminate a session."""
        if not extra:
            session = await self.session_registry.get_session_by_channel(message.channel.id)
            if not session:
                await message.channel.send("Not in a CLI session. Use `!vli exit <id>`.")
                return
            session_id = session.session_id
        else:
            try:
                session_id = int(extra.split()[0])
            except ValueError:
                await message.channel.send("Invalid session ID.")
                return

        session = await self.session_registry.get_session(session_id)
        if not session:
            await message.channel.send(f"Session #{session_id} not found.")
            return

        await message.channel.send(f"Terminating session #{session_id}...")
        await self._cleanup_session(session_id)

    async def _cmd_status(self, message):
        """Show status of current session."""
        session = await self.session_registry.get_session_by_channel(message.channel.id)
        if not session:
            await message.channel.send("Not in a CLI session.")
            return

        embed = discord.Embed(title=f"Session #{session.session_id}", color=0x00ff00)
        embed.add_field(name="Running", value=str(session.is_alive))
        embed.add_field(name="PID", value=str(session.data.pid))
        if session.data.project_name:
            embed.add_field(name="Project", value=session.data.project_name)
        if session.data.workspace_dir:
            embed.add_field(name="Workspace", value=session.data.workspace_dir, inline=False)
        await message.channel.send(embed=embed)

    async def _cmd_attach(self, message):
        """Get command to attach locally."""
        session = await self.session_registry.get_session_by_channel(message.channel.id)
        if not session:
            await message.channel.send("Not in a CLI session.")
            return

        attach_cmd = session.get_attach_command()
        if attach_cmd:
            await message.channel.send(f"**Attach:** `{attach_cmd}`")
        else:
            await message.channel.send("Local attach not available.")

    async def _cmd_resize(self, message, extra: str):
        """Resize terminal."""
        session = await self.session_registry.get_session_by_channel(message.channel.id)
        if not session:
            await message.channel.send("Not in a CLI session.")
            return

        parts = extra.split()
        if len(parts) != 2:
            await message.channel.send("Usage: `!vli resize <cols> <rows>`")
            return

        try:
            cols, rows = int(parts[0]), int(parts[1])
        except ValueError:
            await message.channel.send("Invalid dimensions.")
            return

        if await session.terminal.resize(cols, rows):
            await message.channel.send(f"Resized to {cols}x{rows}")
        else:
            await message.channel.send("Failed to resize.")

    async def _cmd_scroll(self, message, extra: str):
        """Dump scrollback history from terminal."""
        session = await self.session_registry.get_session_by_channel(message.channel.id)
        if not session:
            await message.channel.send("No active session in this channel.")
            return

        lines = 100
        if extra:
            try:
                lines = int(extra.strip())
            except ValueError:
                await message.channel.send("Usage: `!vli scroll [lines]` (default: 100)")
                return

        output = await session.terminal.get_output()
        if not output:
            await message.channel.send("No output available.")
            return

        output_lines = output.split('\n')
        if len(output_lines) > lines:
            output_lines = output_lines[-lines:]

        text = '\n'.join(output_lines)
        chunks = self._split_message(text, 1800)

        await message.channel.send(f"**Scrollback ({len(output_lines)} lines):**")
        for chunk in chunks:
            escaped = chunk.replace('```', '`\u200b`\u200b`')
            await message.channel.send(f"```\n{escaped}\n```")

    async def _cmd_clear(self, message):
        """Clear all messages in CLI channel except the live terminal display."""
        session = await self.session_registry.get_session_by_channel(message.channel.id)
        if not session:
            await message.channel.send("No active session in this channel.")
            return

        live_message_id = None
        if hasattr(session, 'output_manager') and session.output_manager:
            if session.output_manager.message:
                live_message_id = session.output_manager.message.id

        deleted_count = 0
        try:
            async for msg in message.channel.history(limit=200):
                if live_message_id and msg.id == live_message_id:
                    continue
                try:
                    await msg.delete()
                    deleted_count += 1
                except discord.HTTPException:
                    pass

            logger.info(f"Cleared {deleted_count} messages from session {session.session_id} channel")
        except discord.Forbidden:
            await message.channel.send("Missing permission to delete messages.")
        except Exception as e:
            logger.error(f"Error clearing messages: {e}")

    async def _cmd_send_key(self, message, key: str):
        """Send a special key or control character."""
        session = await self.session_registry.get_session_by_channel(message.channel.id)
        if not session:
            await message.channel.send("Not in a CLI session.")
            return

        if not session.is_alive:
            await message.channel.send("Process not running.")
            return

        try:
            await session.terminal.send_control(key)
        except Exception as e:
            await message.channel.send(f"Error: {e}")

    def _split_message(self, text: str, max_len: int = 1900) -> list:
        """Split text into chunks for Discord messages."""
        if len(text) <= max_len:
            return [text]

        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break

            split_at = max_len
            newline_pos = text.rfind('\n', 0, max_len)
            if newline_pos > max_len // 2:
                split_at = newline_pos + 1

            chunks.append(text[:split_at])
            text = text[split_at:]

        return chunks

    # --------------------------------------------------
    # PROJECT COMMANDS (!vp)
    # --------------------------------------------------

    async def _handle_vp_command(self, message, content: str):
        """Route !vp commands."""
        if not self.project_manager:
            await message.channel.send("Project management requires Redis.")
            return

        parts = content.split(None, 2)

        if len(parts) == 1:
            await self._cmd_list_projects(message)
            return

        arg1 = parts[1].strip()
        arg1_lower = arg1.lower()
        extra = parts[2] if len(parts) > 2 else ""

        if arg1_lower == 'ls':
            await self._cmd_list_projects(message)
        elif arg1_lower == 'rm':
            if extra:
                await self._cmd_remove_project(message, extra.split()[0])
            else:
                await message.channel.send("Usage: `!vp rm <name>`")
        else:
            await self._cmd_register_project(message, arg1, extra.strip() if extra else None)

    async def _cmd_list_projects(self, message):
        """List all registered projects."""
        projects = await self.project_manager.list_all()
        if not projects:
            await message.channel.send("No projects registered.")
            return

        embed = discord.Embed(title="Registered Projects", color=0x00ff00)
        for p in projects:
            embed.add_field(name=p.name, value=f"Path: `{p.path}`", inline=False)
        await message.channel.send(embed=embed)

    async def _cmd_register_project(self, message, name: str, path: str = None):
        """Register a new project."""
        import os

        if not path:
            if not self.config.workspace_dir:
                await message.channel.send("No path provided and WORKSPACE_DIR not set.")
                return
            path = os.path.join(self.config.workspace_dir, name)

        path = os.path.expanduser(path)
        path = os.path.abspath(path)

        if not os.path.isdir(path):
            await message.channel.send(f"Path does not exist: `{path}`")
            return

        try:
            project = await self.project_manager.create(name, path)
            await message.channel.send(f"Project `{project.name}` registered at `{project.path}`")
        except ValueError as e:
            await message.channel.send(f"Error: {e}")

    async def _cmd_remove_project(self, message, name: str):
        """Remove a registered project."""
        success = await self.project_manager.delete(name)
        if success:
            await message.channel.send(f"Project `{name}` removed.")
        else:
            await message.channel.send(f"Project `{name}` not found.")


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(AltairCore(bot))
