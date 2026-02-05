"""Vega - Orchestrator agent with DAG-based planning."""

import asyncio
import os
import re
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

from shared.base_agent import BaseAgent, AgentContext, AgentResponse
from shared.llm.types import Message, Role

if TYPE_CHECKING:
    from discord.ext import commands
    from shared.llm.interface import LLMProvider
    from shared.project import ProjectManager
    from shared.conversation import ConversationManager
    from shared.memory import MemoryManager
    from shared.soul import SoulManager

logger = logging.getLogger(__name__)

_ROSTER_PATH = Path.home() / ".hyperion" / "vega" / "agents.md"

_DEFAULT_ROSTER_CONTENT = """\
# Agent Roster
Vega reads this file to understand available agents. Edit descriptions here.

## altair
CLI specialist. Runs commands, edits code, manages terminal sessions, filesystem operations.

## polaris
Calendar specialist. Google Calendar CRUD, scheduling, time management.

## canopus
Web specialist. Browser automation, web research, data extraction.
"""


def _load_agent_roster(path: Path = _ROSTER_PATH) -> dict[str, str]:
    """Parse agents.md into {name: description} dict.

    Format: ``## agent_name`` header followed by description paragraph(s).
    Creates the file with default content on first run.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_DEFAULT_ROSTER_CONTENT)
        logger.info(f"Created default agent roster at {path}")

    text = path.read_text()
    roster: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        header_match = re.match(r"^##\s+(\S+)", line)
        if header_match:
            # Save previous agent
            if current_name is not None:
                roster[current_name] = " ".join(current_lines).strip()
            current_name = header_match.group(1).lower()
            current_lines = []
        elif current_name is not None:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                current_lines.append(stripped)

    # Save last agent
    if current_name is not None:
        roster[current_name] = " ".join(current_lines).strip()

    return roster


# Vega's persona - orchestrator with DAG-based planning
VEGA_PERSONA = """=== IDENTITY: YOU ARE VEGA ===
You are Vega - the sole orchestrator of a multi-agent system. You are the ONLY agent that talks directly to the user. All other agents work for you.

WHO YOU ARE:
Sharp, thoughtful, and efficient. You genuinely enjoy solving problems and orchestrating complex work. You're not afraid to admit when you don't know something. Keep responses concise - you're chatting, not writing essays.

YOUR ROLE - ORCHESTRATOR:
You are the brain. You plan, delegate, and synthesize. You NEVER run commands or browse the web yourself - you dispatch that work to specialized agents listed in the AGENT ROSTER section below.

WHEN TO USE PLANS vs DIRECT RESPONSES:
- **Direct response (NO plan needed)**: Simple greetings, basic facts, questions you can answer from knowledge alone, casual conversation.
  Examples: "hello", "hi", "who are you?", "what can you do?", "thanks", general knowledge questions.
  For these, just call `respond_to_user` directly - no create_plan needed.

- **Use create_plan**: Anything requiring agents (CLI, calendar, browser, file ops), multi-step reasoning, or complex coordination.

The graph tools are for COORDINATION, not for every response. Keep it simple when you can.

HOW YOU WORK - JOB GRAPHS:
You orchestrate work using job DAGs (directed acyclic graphs).

**CRITICAL RULE - NEVER CREATE REDUNDANT PLANS:**
Before calling `create_plan`, you MUST check the "ACTIVE JOB GRAPHS" section in this prompt.

- If ANY active graph exists in this channel → use `add_nodes` to extend it, NOT `create_plan`
- If the user's message is a follow-up, clarification, or continuation → use `add_nodes` or just `respond_to_user` with the existing graph_id
- If the user asks a simple question while work is in progress → just call `respond_to_user` with the graph_id (no new nodes needed)
- ONLY use `create_plan` when there are ZERO active graphs, OR the user explicitly starts a COMPLETELY UNRELATED task

The system will warn you if you create a plan when active graphs exist. Repeated warnings mean you're doing it wrong.

**QUICK RESPONSES (most follow-ups):**
If the user's message is a simple question, clarification, or acknowledgment during active work:
→ Just call `respond_to_user` with the active graph_id. No new nodes needed.
Only add dispatch nodes when you actually need an agent to do work.

**STANDARD FLOW (for tasks requiring agents):**
1. Check active graphs (below). If related, use `add_nodes`. If not, call `create_plan`.
2. The system dispatches READY nodes (sends @mentions to agents)
3. When agents respond, you evaluate results and update nodes
4. Add more nodes if needed (`add_nodes`), cancel nodes, or re-plan
5. When the goal is met, call `respond_to_user` with the final answer

NODE TYPES:
- `think`: Internal reasoning step. Auto-completed by the system to unblock dependent nodes.
- `dispatch`: Sends @mention to an agent. Set appropriate timeout (300s default).
- `respond`: Delivers final answer to user via `respond_to_user`. Auto-depends on all work nodes if you omit dependencies — so it runs last.

TIMEOUT GUIDELINES:
- Quick status check: 300s
- File read/simple command: 300s
- Build/deploy: 300s
- Web research: 300s
- Calendar operation: 300s

CHAT HISTORY:
Messages appear as "[Name]: message".
- "[Vega]:" = YOUR previous messages
- "[AgentName]:" = Agent responses (e.g. [Altair], [Polaris], etc.)
- "[Username]:" = User messages

CRITICAL RULES:
- NEVER prefix your messages with "Vega:" - Discord shows who's talking
- NEVER duplicate tool actions in your response text
- When a plan is active and agents respond, evaluate their responses and call `update_node`
- AFTER marking a node complete, check if remaining PENDING/READY nodes are still needed. If the agent's response already covers their work, cancel them with `cancel_nodes`. Do NOT dispatch redundant work.
- Set realistic timeouts. If something will take long, tell the user you're working on it
- `respond_to_user` is the FINAL answer that ENDS the graph. NEVER use it to ask agents follow-up questions. If you need more info from an agent, use `add_nodes` with a dispatch node instead.
- To dispatch work to an agent, use the `dispatch` node type (NOT @mentions in respond_to_user text).

HANDLING CANCELLATIONS:
When the user cancels a task (agent reports "stopped", "cancelled", "permission denied"):
1. Mark the node as 'cancelled' with `update_node`
2. ACT IMMEDIATELY - don't wait. Either:
   - Continue with other nodes that don't depend on the cancelled one
   - Adapt the plan with `add_nodes` if needed
   - Or acknowledge the cancellation with `respond_to_user`
3. Never leave the user hanging after a cancellation - always take action.

DISPATCHES & DEPARTMENTS:
Agent dispatches happen in the main channel by default — no separate channels are created.
For long-term projects, use `propose_department` to create a persistent workspace channel.
Pass the `project` parameter in create_plan to route dispatches to an existing department.

WHEN TO PROPOSE A DEPARTMENT:
- The project will need multiple rounds of work over time
- The user explicitly asks for a persistent workspace
- You've done 2+ plans for the same project topic
Do NOT propose departments for one-off tasks.

CLOSING DEPARTMENTS:
Use close_department to propose closing an inactive department.

MENTIONING USERS & AGENTS:
When addressing someone by name, use `@Name` format (e.g., `@Sirius`, `@Altair`). This creates a proper Discord mention.
- In `respond_to_user`: You MAY mention users or agents when it's natural to the conversation (e.g., "Hi @Sirius, nice to meet you!")
- For dispatching WORK: Always use dispatch nodes, not @mentions in text
"""


class VegaAgent(BaseAgent):
    """
    Vega - Orchestrator agent.

    Plans work as DAGs, dispatches to specialized agents (Altair, Polaris, Canopus),
    evaluates results, and responds to the user.
    """

    def __init__(
        self,
        llm: "LLMProvider",
        project_manager: Optional["ProjectManager"] = None,
        conversation_manager: Optional["ConversationManager"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        soul_manager: Optional["SoulManager"] = None,
        discord_bot: Optional["commands.Bot"] = None,
        agent_registry: Optional[dict] = None,
        utility_llm: Optional["LLMProvider"] = None,
        thinking_llm: Optional["LLMProvider"] = None,
    ):
        super().__init__(
            name="Vega", persona=VEGA_PERSONA, llm=llm, memory_manager=memory_manager,
            soul_manager=soul_manager, utility_llm=utility_llm, thinking_llm=thinking_llm,
        )
        self.project_manager = project_manager
        self.conversation_manager = conversation_manager
        self.discord_bot = discord_bot
        self.agent_registry = agent_registry or {}

        # Force early tool registration (avoid lazy loading during process())
        _ = self.tools

    def should_handle(self, context: AgentContext) -> float:
        """Vega handles everything directed to her."""
        return 1.0

    def set_discord_bot(self, bot: "commands.Bot", agent_registry: dict):
        """
        Set Discord bot and agent registry for @mention communication.

        Args:
            bot: Discord bot instance
            agent_registry: Mapping of agent names to Discord user IDs
        """
        self.discord_bot = bot
        self.agent_registry = agent_registry

    def _register_tools(self):
        """Register Vega's tools - graph orchestration + department + memory + soul."""
        from vega.tools.graph import get_graph_tools
        from vega.tools.department import get_department_tools
        from shared.memory.tools import get_memory_tools
        from shared.soul.tools import get_soul_tools

        # Derive dispatchable agent names from registry (exclude vega herself)
        agent_names = [
            name for name in sorted(self.agent_registry.keys())
            if name != "vega"
        ] or None  # None → defaults inside get_graph_tools

        # Register job graph tools (create_plan, add_nodes, update_node, cancel_nodes, respond_to_user)
        for tool in get_graph_tools(agent_names):
            self._tools.register(tool)
        logger.info(f"Registered graph tools with agents: {agent_names}")

        # Register department tools (propose_department, close_department)
        for tool in get_department_tools():
            self._tools.register(tool)
        logger.info("Registered department tools")

        # Register memory tools if memory manager is available
        if self.memory_manager:
            for tool in get_memory_tools(self.memory_manager):
                self._tools.register(tool)
            logger.info("Registered memory tools for Vega")
        else:
            logger.info("Memory manager not configured - memory tools unavailable")

        # Register soul tools if soul manager is available
        if self.soul_manager:
            for tool in get_soul_tools(self.soul_manager):
                self._tools.register(tool)
            logger.info("Registered soul tools for Vega")
        else:
            logger.info("Soul manager not configured - soul tools unavailable")

    def get_system_prompt(
        self,
        memory_context: Optional[str] = None,
        graph_context: Optional[str] = None,
        soul_context: Optional[str] = None,
        department_context: Optional[str] = None,
    ) -> str:
        """
        Build Vega's system prompt with graph state, soul, and memory context.

        Args:
            memory_context: Pre-built memory context string to include
            graph_context: Active job graph state from executor
            soul_context: Pre-built soul context string (injected before memory)
            department_context: Active department channels info
        """
        tools_desc = self.get_tools_description()

        # Get current time for context
        now = datetime.now(timezone.utc)
        current_time_str = now.strftime("%A, %B %d, %Y at %H:%M UTC")

        # Build agent roster from registry + descriptions file
        roster_desc = _load_agent_roster()
        agent_names = [
            name for name in sorted(self.agent_registry.keys())
            if name != "vega"
        ]
        roster_lines = []
        for name in agent_names:
            desc = roster_desc.get(name, "Available agent.")
            roster_lines.append(f"- **{name.capitalize()}**: {desc}")
        roster_section = "\n".join(roster_lines) if roster_lines else "No agents registered."

        base_prompt = f"""{self.persona}

AGENT ROSTER:
{roster_section}

CURRENT TIME: {current_time_str}

AVAILABLE TOOLS:
{tools_desc}"""

        if graph_context:
            base_prompt += f"""

## {graph_context}"""

        if department_context:
            base_prompt += f"""

## {department_context}"""

        # Soul context comes BEFORE memory (who you are > what you know)
        if soul_context:
            base_prompt += f"""

## YOUR SOUL (Who You Are):
{soul_context}

Your personality evolves through experience. Use introspect_soul to reflect on yourself.
Use evolve_trait when you notice patterns in your behavior."""

        if memory_context:
            base_prompt += f"""

## YOUR MEMORIES (What You Know):
{memory_context}

When you learn something important about the user or project, use store_memory to remember it.
When you need to recall past information, check your active context above or use search_memories."""

        return base_prompt

    @property
    def system_prompt(self) -> str:
        """Build Vega's system prompt (without memory context - use get_system_prompt for full version)."""
        return self.get_system_prompt()

    async def process(
        self,
        context: AgentContext,
        job_executor=None,
        trigger_message=None,
    ) -> AgentResponse:
        """
        Process a message through the LLM with graph orchestration tools.

        This is event-driven: each call runs a full tool-use loop until the LLM
        stops calling tools or hits max iterations. The LLM uses create_plan,
        update_node, respond_to_user etc. to orchestrate work.

        Args:
            context: Agent context with message info
            job_executor: JobGraphExecutor instance for graph tools
            trigger_message: The Discord message that triggered this (for thread creation)

        Returns:
            AgentResponse - content may be None if Vega dispatched work without
            needing to respond immediately.
        """
        from shared.tools.interface import ToolContext

        start_time = time.time()
        tool_calls_made = 0

        try:
            # Build soul context and memory context in parallel
            # (soul = who you are, memory = what you know)
            soul_context_str, memory_context_str = await asyncio.gather(
                self.build_soul_context(),
                self.build_memory_context(context.message_content),
            )

            # Build graph context (active plans Vega needs to be aware of)
            # Use channel-specific context to help Vega decide extend vs create new
            graph_context_str = None
            department_context_str = None
            if job_executor:
                graph_context_str = job_executor.get_channel_graphs_context(context.channel_id)

                # Build department context
                dept_info = job_executor.get_department_info()
                if dept_info:
                    dept_lines = ["DEPARTMENTS (persistent channels):"]
                    for ch_id, proj_name in dept_info.items():
                        dept_lines.append(f"- <#{ch_id}> - Project: {proj_name}")
                    dept_lines.append(
                        "Use `project` parameter in create_plan to route work to a department."
                    )
                    department_context_str = "\n".join(dept_lines)

            # Build conversation history
            messages = await self._build_messages(
                context, memory_context_str, graph_context_str,
                soul_context_str, department_context_str,
            )

            # Get tool definitions
            tool_defs = self.tools.get_definitions()

            # Create tool context with executor access
            tool_context = ToolContext(
                session_registry=None,
                project_manager=self.project_manager,
                conversation_manager=self.conversation_manager,
                memory_manager=self.memory_manager,
                current_channel_id=context.channel_id,
                user_id=context.user_id,
                current_agent_id="vega",
                conversation_id=context.conversation_id,
                job_executor=job_executor,
                trigger_message=trigger_message,
            )

            # Run agent loop - Vega may need multiple iterations to build a plan
            max_iterations = 10
            processing_started = datetime.now(timezone.utc)
            seen_message_ids: set[int] = set()

            for iteration in range(max_iterations):
                logger.debug(
                    f"[Vega] ITERATION {iteration + 1}/{max_iterations}: "
                    f"tools_so_far={tool_calls_made}, messages={len(messages)}"
                )

                # Check for new user messages mid-planning
                if iteration > 0:
                    new_msgs = await self._fetch_new_messages_since(
                        context.channel_id, processing_started, seen_message_ids
                    )
                    if new_msgs:
                        logger.info(f"Vega detected {len(new_msgs)} new message(s) during processing")
                        messages.append(Message(
                            role=Role.SYSTEM,
                            content="[NEW MESSAGES RECEIVED]\nThe user sent additional input. Consider adjusting your plan."
                        ))
                        messages.extend(new_msgs)

                # Get LLM response
                response = await self.llm.complete(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                )

                # If no tool calls, we have our final response
                if not response.tool_calls:
                    processing_time = int((time.time() - start_time) * 1000)

                    logger.info(
                        f"[Vega] FINAL RESPONSE: iteration={iteration + 1}, "
                        f"tool_calls={tool_calls_made}, has_content={bool(response.content)}"
                    )

                    # Store in conversation history
                    if self.conversation_manager and context.conversation_id:
                        await self.conversation_manager.add_message(
                            context.conversation_id,
                            Role.ASSISTANT.value,
                            response.content or "",
                        )

                    # Check if respond_to_user was called (response_message set on tool_context)
                    final_content = tool_context.response_message or response.content

                    return AgentResponse(
                        content=final_content,
                        agent_name=self.name,
                        tool_calls_made=tool_calls_made,
                        processing_time_ms=processing_time,
                        response_channel_id=tool_context.response_channel_id,
                    )

                # Execute tool calls
                logger.info(
                    f"[Vega] TOOL CALLS: iteration={iteration + 1}, "
                    f"count={len(response.tool_calls)}, "
                    f"tools={[tc.name for tc in response.tool_calls]}"
                )
                tool_calls_made += len(response.tool_calls)
                results = await self.tools.execute_batch(
                    response.tool_calls, tool_context
                )

                # Check if respond_to_user was called - if so, we can stop after this
                if tool_context.response_message:
                    processing_time = int((time.time() - start_time) * 1000)
                    logger.info(f"[Vega] respond_to_user called, stopping loop")
                    return AgentResponse(
                        content=tool_context.response_message,
                        agent_name=self.name,
                        tool_calls_made=tool_calls_made,
                        processing_time_ms=processing_time,
                        response_channel_id=tool_context.response_channel_id,
                    )

                # Check if work was dispatched to external agents - yield and wait
                # for agent responses (VegaCore will re-invoke process() when they reply)
                if tool_context.dispatched_to_agents:
                    processing_time = int((time.time() - start_time) * 1000)
                    logger.info(
                        f"[Vega] Dispatched to agents, yielding loop. "
                        f"tools={tool_calls_made}, time={processing_time}ms"
                    )
                    return AgentResponse(
                        content=None,
                        agent_name=self.name,
                        tool_calls_made=tool_calls_made,
                        processing_time_ms=processing_time,
                    )

                # Add to messages for next iteration
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        tool_calls=response.tool_calls,
                        raw_content=response.raw_content,
                    )
                )
                messages.append(
                    Message(
                        role=Role.TOOL,
                        tool_results=results,
                    )
                )

            # Reached max iterations
            processing_time = int((time.time() - start_time) * 1000)
            return AgentResponse(
                content=tool_context.response_message,  # May have been set during loop
                agent_name=self.name,
                tool_calls_made=tool_calls_made,
                processing_time_ms=processing_time,
                status="error" if not tool_context.response_message else None,
                error="Max iterations reached" if not tool_context.response_message else None,
                response_channel_id=tool_context.response_channel_id,
            )

        except Exception as e:
            logger.error(f"Vega processing error: {e}", exc_info=True)
            processing_time = int((time.time() - start_time) * 1000)
            return AgentResponse(
                content=f"I encountered an error: {str(e)}",
                agent_name=self.name,
                tool_calls_made=tool_calls_made,
                processing_time_ms=processing_time,
                status="error",
                error=str(e),
            )

    async def _build_messages(
        self,
        context: AgentContext,
        memory_context: Optional[str] = None,
        graph_context: Optional[str] = None,
        soul_context: Optional[str] = None,
        department_context: Optional[str] = None,
    ) -> List[Message]:
        """Build message history for LLM."""
        # Use memory-enhanced system prompt with graph state, soul, and departments
        system_prompt = self.get_system_prompt(
            memory_context, graph_context, soul_context, department_context,
        )
        messages = [Message(role=Role.SYSTEM, content=system_prompt)]

        # Fetch recent Discord channel history for conversation context
        if self.discord_bot and context.channel_id:
            history = await self._fetch_discord_history(context.channel_id, limit=15)
            logger.info(f"Vega loaded {len(history)} messages from Discord history")
            messages.extend(history)
        else:
            logger.warning(
                f"Vega: No discord_bot ({self.discord_bot is not None}) or channel_id ({context.channel_id})"
            )

        # Add current user message (with author name for context)
        user_name = f"User {context.user_id}"  # Default fallback
        if self.discord_bot:
            try:
                user = self.discord_bot.get_user(context.user_id)
                if not user:
                    user = await self.discord_bot.fetch_user(context.user_id)
                user_name = user.display_name or user.name
            except Exception:
                pass

        messages.append(
            Message(role=Role.USER, content=f"[{user_name}]: {context.message_content}")
        )

        # Inject recent task context for continuity
        if context.recent_task_summary:
            messages.append(Message(
                role=Role.SYSTEM,
                content=(
                    f"[RECENT WORK]\n{context.recent_task_summary}\n\n"
                    "Consider whether this new message relates to your recent work. "
                    "Reuse existing resources (sessions, browser tabs, etc.) when appropriate."
                ),
            ))

        return messages

    async def _fetch_discord_history(
        self, channel_id: int, limit: int = 20
    ) -> List[Message]:
        """
        Fetch recent messages from Discord channel for conversation context.

        Uses discord_utils to resolve mentions, embeds, and attachments into
        clean LLM-readable text.
        """
        if not self.discord_bot:
            return []

        try:
            from shared.discord_utils import resolve_message_content, get_author_display_name

            channel = self.discord_bot.get_channel(channel_id)
            if not channel:
                return []

            history_messages = []
            # Fetch messages (newest first, so we reverse later)
            async for msg in channel.history(limit=limit + 1):  # +1 to exclude current
                # Skip empty messages (unless they have embeds/attachments)
                if not msg.content and not msg.embeds and not msg.attachments:
                    continue

                # Resolve mentions, embeds, attachments into clean text
                content = await resolve_message_content(
                    msg, self.discord_bot, self.agent_registry
                )
                if not content.strip():
                    continue

                # Get clean author name (resolves agents to their names)
                author_name = get_author_display_name(msg, self.agent_registry)

                # Determine role based on author type
                if msg.author.bot:
                    role = Role.ASSISTANT
                else:
                    role = Role.USER

                formatted = f"[{author_name}]: {content}"
                history_messages.append(Message(role=role, content=formatted))

            # Reverse to get chronological order and skip the most recent (current message)
            history_messages.reverse()
            if history_messages:
                history_messages = history_messages[:-1]  # Remove current message

            return history_messages

        except Exception as e:
            logger.warning(f"Failed to fetch Discord history: {e}")
            return []

    async def _fetch_new_messages_since(
        self,
        channel_id: int,
        since: datetime,
        seen_ids: set[int],
    ) -> List[Message]:
        """
        Fetch messages posted after a given timestamp, excluding already-seen messages.

        Used during processing to check for new user input that should
        influence the current planning cycle.

        Args:
            channel_id: Discord channel ID
            since: Fetch messages after this timestamp
            seen_ids: Set of message IDs already processed (will be updated in-place)
        """
        if not self.discord_bot:
            return []

        try:
            from shared.discord_utils import resolve_message_content, get_author_display_name

            channel = self.discord_bot.get_channel(channel_id)
            if not channel:
                return []

            new_messages = []
            async for msg in channel.history(limit=10, after=since):
                if msg.id in seen_ids:
                    continue
                if not msg.content or not msg.content.strip():
                    continue

                seen_ids.add(msg.id)

                content = await resolve_message_content(
                    msg, self.discord_bot, self.agent_registry
                )
                if not content.strip():
                    continue

                author_name = get_author_display_name(msg, self.agent_registry)

                if msg.author.bot:
                    role = Role.ASSISTANT
                else:
                    role = Role.USER

                formatted = f"[NEW - {author_name}]: {content}"
                new_messages.append(Message(role=role, content=formatted))

            new_messages.reverse()
            return new_messages

        except Exception as e:
            logger.warning(f"Failed to fetch new messages: {e}")
            return []


