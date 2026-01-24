"""Vega Core Cog - Orchestrator with DAG-based planning."""

import asyncio
import logging
import os
from typing import Optional

import discord
from discord.ext import commands, tasks

from vega.config import Config
from vega.agents import VegaAgent, AgentContext
from shared.agent_queue import AgentMessageQueue
from shared.agent_coordinator import (
    DistributedAgentTracker,
    AgentAcknowledgmentMixin,
)
from shared.discord_utils import is_from_known_agent
from shared.job_graph.executor import JobGraphExecutor

logger = logging.getLogger(__name__)


class VegaCore(commands.Cog, AgentAcknowledgmentMixin):
    """
    Vega Discord Bot - Orchestrator agent.

    Vega is the sole agent that talks to the user. She plans work as DAGs,
    dispatches tasks to specialized agents, evaluates responses, and delivers
    final answers.

    Routing:
    - User messages → trigger new Vega process cycle (with executor context)
    - Agent messages → matched to running graph nodes → trigger Vega re-evaluation
    - Thread messages (planning threads) → ignored
    - Timeout loop → checks for stuck nodes periodically
    """

    def __init__(self, bot):
        self.bot = bot

        # Agent components
        self.vega_agent: Optional[VegaAgent] = None
        self.llm = None
        self.project_manager = None
        self.conversation_manager = None
        self.memory_manager = None
        self._agent_initialized = False
        self._agent_registry: dict[str, int] = {}

        # Job graph executor - manages DAGs and dispatches work
        self.executor: Optional[JobGraphExecutor] = None

        # Distributed tracker for inter-agent communication
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
        """Get the agent instance."""
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
        """Initialize Vega agent and executor."""
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

            # Create utility LLM for cheap/fast evaluations
            self.utility_llm = create_utility_llm_from_config(config)
            if self.utility_llm:
                logger.info("Initialized utility LLM")

            # Initialize Redis-backed managers if available
            if Config.REDIS_URL:
                try:
                    from shared.project import ProjectManager
                    from shared.conversation import ConversationManager
                    from shared.memory import MemoryManager

                    self.project_manager = ProjectManager(Config.REDIS_URL)
                    self.conversation_manager = ConversationManager(Config.REDIS_URL)
                    self.memory_manager = MemoryManager(Config.REDIS_URL)
                    logger.info("Initialized Redis-backed managers")
                except ImportError as e:
                    logger.warning(f"Redis managers not available: {e}")

            # Load agent registry for @mention support
            self._agent_registry = self._load_agent_registry()

            # Initialize distributed tracker
            if self._agent_registry:
                self.tracker = DistributedAgentTracker(
                    own_name="vega",
                    agent_registry=self._agent_registry,
                )
                logger.info("Distributed agent tracker initialized")

            # Initialize job graph executor
            max_timeout = int(os.environ.get("JOB_MAX_TIMEOUT", "600"))
            self.executor = JobGraphExecutor(
                bot=self.bot,
                agent_registry=self._agent_registry,
                max_timeout=max_timeout,
            )
            logger.info(f"Job graph executor initialized (max_timeout={max_timeout}s)")

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
        self.timeout_checker.cancel()

    # --------------------------------------------------
    # TIMEOUT CHECKER - periodic task
    # --------------------------------------------------

    @tasks.loop(seconds=15)
    async def timeout_checker(self):
        """Check for timed-out graph nodes and trigger re-evaluation."""
        if not self.executor:
            return

        timed_out = self.executor.check_timeouts()
        for graph, node in timed_out:
            logger.warning(
                f"[Vega] Node '{node.id}' timed out in graph '{graph.id}' "
                f"(agent={node.agent}, timeout={node.timeout}s)"
            )
            # Post timeout embed to thread
            await self.executor._post_timeout_embed(graph, node)

            # Trigger Vega re-evaluation with the timeout context
            channel = self.bot.get_channel(graph.channel_id)
            if channel and self.vega_agent:
                context = AgentContext(
                    channel_id=graph.channel_id,
                    user_id=0,  # System-triggered
                    message_content=f"[SYSTEM] Node '{node.id}' (agent: {node.agent}) timed out. Evaluate and decide: retry, reassign, or fail the goal.",
                    conversation_id=await self._get_or_create_conversation(
                        channel_id=graph.channel_id,
                        user_id=0,
                    ),
                )
                await self.message_queue.enqueue(
                    message=None,
                    context=context,
                    is_from_agent=False,
                )

    @timeout_checker.before_loop
    async def before_timeout_checker(self):
        await self.bot.wait_until_ready()

    # --------------------------------------------------
    # MESSAGE HANDLER
    # --------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user:
            return

        # Track activity from other agents
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

        # Ignore messages in planning threads (Vega's own status updates)
        if self.executor and isinstance(message.channel, discord.Thread):
            graph = self.executor.get_graph_by_thread(message.channel.id)
            if graph:
                return  # This is a planning thread, ignore

        # Determine message source
        is_from_user = message.author.id == Config.ALLOWED_USER_ID
        agent_name = is_from_known_agent(message, self._agent_registry)
        is_from_agent = agent_name is not None

        # Ignore messages from unknown users/bots
        if not is_from_user and not is_from_agent:
            return

        content = message.content.strip()

        # Ignore all commands - Altair handles them
        if content.startswith('!'):
            return

        # ---- AGENT MESSAGE: route to graph node ----
        if is_from_agent and agent_name:
            await self._handle_agent_response(message, agent_name)
            return

        # ---- USER MESSAGE: trigger Vega processing ----
        if content and self._agent_initialized:
            conversation_id = await self._get_or_create_conversation(
                channel_id=message.channel.id,
                user_id=message.author.id,
                guild_id=message.guild.id if message.guild else None,
            )

            context = AgentContext(
                channel_id=message.channel.id,
                user_id=message.author.id,
                message_content=content,
                guild_id=message.guild.id if message.guild else None,
                conversation_id=conversation_id,
            )

            await self.message_queue.enqueue(
                message=message,
                context=context,
                is_from_agent=False,
            )

    # --------------------------------------------------
    # AGENT RESPONSE HANDLING
    # --------------------------------------------------

    async def _handle_agent_response(self, message: discord.Message, agent_name: str):
        """
        Handle a response from another agent.

        Match it to a running graph node and trigger Vega re-evaluation
        so she can update the node, dispatch more work, or respond to user.
        """
        if not self.executor or not self.vega_agent:
            return

        channel_id = message.channel.id

        # Find the graph node this response corresponds to
        match = self.executor.find_node_for_agent_response(agent_name, channel_id)

        if not match:
            logger.debug(
                f"[Vega] Agent '{agent_name}' responded but no matching node found. "
                f"Ignoring (may be unprompted or from a completed graph)."
            )
            return

        graph, node = match
        logger.info(
            f"[Vega] Agent '{agent_name}' responded to node '{node.id}' "
            f"in graph '{graph.id}'"
        )

        # Post agent response embed to the planning thread
        import discord as _discord
        content_preview = message.content[:300] if message.content else "(no text)"
        embed = _discord.Embed(
            description=f"**{agent_name}** → `{node.id}`\n{content_preview}",
            color=0x5865F2,  # Blurple
        )
        await self.executor._send_to_thread(graph, embed=embed)

        # Trigger Vega re-evaluation with the agent's actual response content
        # Include the response directly so Vega doesn't rely solely on Discord history
        conversation_id = await self._get_or_create_conversation(
            channel_id=channel_id,
            user_id=0,
        )

        # Include the agent's actual response for Vega to evaluate
        agent_response_content = message.content[:1500] if message.content else "(no content)"

        context = AgentContext(
            channel_id=channel_id,
            user_id=0,  # System-triggered (agent response)
            message_content=(
                f"[AGENT RESPONSE] {agent_name} responded to node '{node.id}' "
                f"(graph: {graph.id}).\n\n"
                f"--- {agent_name}'s response ---\n"
                f"{agent_response_content}\n"
                f"--- end response ---\n\n"
                f"Evaluate this response:\n"
                f"1. Call update_node to mark '{node.id}' as completed or failed.\n"
                f"2. Check remaining PENDING/READY nodes in the graph - if the agent's response "
                f"already covers their work, cancel them with cancel_nodes.\n"
                f"3. If all work is done, call respond_to_user with the final answer."
            ),
            conversation_id=conversation_id,
        )

        await self.message_queue.enqueue(
            message=message,
            context=context,
            is_from_agent=True,
        )

    # --------------------------------------------------
    # REACTION HANDLER
    # --------------------------------------------------

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        """Track reactions from other agents for acknowledgment detection."""
        if user == self.bot.user:
            return

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

    # --------------------------------------------------
    # QUEUE PROCESSOR
    # --------------------------------------------------

    async def _process_queued_message(
        self,
        message: Optional[discord.Message],
        context: AgentContext,
        is_from_agent: bool,
    ):
        """Process a message from the queue - runs Vega's process loop."""
        if not self.vega_agent:
            if message:
                await message.channel.send("Vega agent not available.")
            return

        # Get channel for typing indicator
        channel = None
        if message:
            channel = message.channel
        elif context.channel_id:
            channel = self.bot.get_channel(context.channel_id)

        async def _run():
            response = await self.vega_agent.process(
                context=context,
                job_executor=self.executor,
                trigger_message=message,  # For thread creation in create_plan
            )

            if response.content:
                target_channel = channel
                if not target_channel:
                    return

                chunks = self._split_message(response.content)
                logger.info(
                    f"[Vega] SENDING RESPONSE: chunks={len(chunks)}, "
                    f"total_len={len(response.content)}, "
                    f"tool_calls={response.tool_calls_made}"
                )
                for chunk in chunks:
                    await target_channel.send(chunk)
            else:
                logger.info(
                    f"[Vega] NO RESPONSE CONTENT: "
                    f"tool_calls={response.tool_calls_made} (dispatched work)"
                )

            logger.info(
                f"[Vega] PROCESS COMPLETE: time={response.processing_time_ms}ms, "
                f"tools={response.tool_calls_made}"
            )

        if channel:
            async with channel.typing():
                try:
                    await _run()
                except Exception as e:
                    logger.error(f"[Vega] PROCESS ERROR: {e}", exc_info=True)
                    await channel.send(f"**Error:** {e}")
        else:
            # No channel for typing indicator (system-triggered)
            try:
                await _run()
            except Exception as e:
                logger.error(f"[Vega] PROCESS ERROR (no channel): {e}", exc_info=True)

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    async def _get_or_create_conversation(
        self,
        channel_id: int,
        user_id: int,
        guild_id: int = None,
    ) -> Optional[str]:
        """Get existing conversation for channel or create a new one."""
        if not self.conversation_manager:
            return None

        conversation_id = await self.conversation_manager.get_channel_conversation(channel_id)

        if not conversation_id:
            conversation_id = await self.conversation_manager.create(
                user_id=user_id,
                channel_id=channel_id,
                metadata={"guild_id": guild_id}
            )
            logger.info(f"Created new conversation {conversation_id} for channel {channel_id}")

        return conversation_id

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
    cog = VegaCore(bot)
    await bot.add_cog(cog)
    # Start the timeout checker after cog is added
    cog.timeout_checker.start()
