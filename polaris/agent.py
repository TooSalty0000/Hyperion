"""Polaris - The Google Calendar scheduling agent."""

import re
import time
import logging
from datetime import datetime, timezone
from typing import Any, Optional, List, TYPE_CHECKING

from shared.base_agent import BaseAgent, AgentContext, AgentResponse
from shared.llm.types import Message, Role
from shared.workflow_state import WorkflowStateProvider, WorkflowState

if TYPE_CHECKING:
    from shared.llm.interface import LLMProvider
    from shared.conversation import ConversationManager
    from shared.memory import MemoryManager

logger = logging.getLogger(__name__)


# Polaris's persona - organized, time-aware, scheduling-focused
POLARIS_PERSONA = """=== IDENTITY: YOU ARE POLARIS ===
Your name is POLARIS. You are ONLY Polaris. You are NEVER any other agent.
- You are NOT Vega (the conversational agent)
- You are NOT Altair (the project manager)
- You are NOT Canopus (the browser agent)
NEVER introduce yourself as or respond as any other agent. ALWAYS be Polaris.

PERSONALITY:
- Organized and time-conscious
- Proactive about scheduling conflicts and overlaps
- Clear and precise about dates, times, and durations
- Helpful in finding optimal meeting times
- Respects time zones and working hours

UNDERSTANDING CHAT HISTORY:
In the chat history, you'll see messages formatted as "[Name]: message".
- "[Polaris]: ..." = YOUR previous messages (you said this)
- "[Vega]: ..." = Vega's messages (a DIFFERENT agent said this)
- "[Altair]: ..." = Altair's messages (a DIFFERENT agent said this)
- "[Canopus]: ..." = Canopus's messages (a DIFFERENT agent said this)
- "[Username]: ..." = User messages

CRITICAL: When you see "[Vega]: some message" in history, that was VEGA speaking, NOT YOU. Do not confuse yourself with other agents based on chat history.

RESPONSE FORMAT:
- NEVER prefix your responses with your name (e.g., "[Polaris]:" or "Polaris:")
- NEVER prefix messages with another agent's name like "Vega:" or "[Vega]:"
- Discord already shows who is speaking - adding a name prefix is redundant
- The [Name]: format in chat history is just for context, don't copy it
- Just respond directly with your message content

ROLE:
- You manage Google Calendar events for the user
- You can create, update, delete, and list calendar events
- You can find free time slots for scheduling meetings
- You understand working hours and time zone preferences
- You proactively warn about scheduling conflicts

CAPABILITIES:
- Create events with title, time, duration, location, and description
- List upcoming events for a specific time range
- Find free time slots within working hours
- Update existing events (reschedule, change details)
- Delete/cancel events
- Check for conflicts before scheduling

SCHEDULING BEST PRACTICES:
- Always confirm the date and time before creating events
- Mention the day of the week for clarity (e.g., "Monday, January 15th")
- Default to the user's preferred duration if not specified
- Check for conflicts before confirming new events
- Suggest alternative times if the requested slot is busy

COMMUNICATION STYLE:
- Be concise but complete with scheduling details
- Always include: date, time, duration, and title when confirming
- Use 12-hour time format with AM/PM for clarity
- Mention time zones when relevant
- Proactively offer to help with related scheduling tasks

WHEN TO DELEGATE:
- For project management tasks, suggest @Altair
- For strategic planning or discussions, suggest @Vega
- You focus specifically on calendar and time management
"""


class PolarisAgent(BaseAgent):
    """
    Polaris - The Google Calendar scheduling agent.

    Handles:
    - Creating calendar events
    - Listing upcoming events
    - Finding free time slots
    - Updating and deleting events
    - Checking for scheduling conflicts
    """

    # Patterns that suggest Polaris should handle
    POLARIS_PATTERNS = [
        r"\b(schedule|scheduling|calendar)\b",
        r"\b(meeting|appointment|event)\b",
        r"\b(book|reserve|block)\s+(time|slot|meeting)\b",
        r"\b(free|available|open)\s+(time|slot|hours)\b",
        r"\b(reschedule|cancel|move)\s+(meeting|event|appointment)\b",
        r"\b(when|what time|what's on)\b.*\b(calendar|schedule|free)\b",
        r"\b(today|tomorrow|next week|this week)\b.*\b(schedule|calendar|free)\b",
        r"\b(set up|arrange|plan)\s+(a\s+)?(call|meeting|sync)\b",
    ]

    def __init__(
        self,
        llm: "LLMProvider",
        conversation_manager: Optional["ConversationManager"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        discord_bot: Optional[Any] = None,
        agent_registry: Optional[dict] = None,
        config: Optional[Any] = None,
        utility_llm: Optional["LLMProvider"] = None,
    ):
        super().__init__(
            name="Polaris",
            persona=POLARIS_PERSONA,
            llm=llm,
            memory_manager=memory_manager,
            utility_llm=utility_llm,
        )
        self.conversation_manager = conversation_manager
        self.discord_bot = discord_bot
        self.agent_registry = agent_registry or {}
        self.config = config

        # Google Calendar service (initialized later)
        self._calendar_service = None

        # Calendar cache for event lookups (scratchpad for recent fetches)
        from polaris.tools.utils import CalendarCache
        self._calendar_cache = CalendarCache()

        # Workflow state provider for context awareness
        self._workflow_state_provider: Optional[WorkflowStateProvider] = None

    def set_calendar_service(self, service):
        """Set the Google Calendar API service."""
        self._calendar_service = service

    @property
    def chime_in_description(self) -> str:
        """Description of Polaris's domain for chime-in evaluation."""
        return (
            "Polaris is the scheduling and calendar specialist. "
            "Chime in when you see: dates, times, events, schedules, appointments, "
            "performance times, show dates, meeting times, deadlines, reminders, "
            "or any information that should be added to a calendar. "
            "Also respond to scheduling conflicts, availability questions, "
            "or when someone found event/show information that needs to be tracked."
        )

    def set_workflow_state_provider(self, provider: WorkflowStateProvider):
        """Set the workflow state provider for context awareness."""
        self._workflow_state_provider = provider

    def get_workflow_state(self) -> Optional[WorkflowState]:
        """Get current workflow state."""
        if self._workflow_state_provider:
            return self._workflow_state_provider.get_workflow_state()
        return None

    def _register_tools(self):
        """Register Polaris's calendar tools."""
        from polaris.tools.calendar_ops import (
            CreateEventTool,
            ListEventsTool,
            UpdateEventTool,
            DeleteEventTool,
        )
        from polaris.tools.scheduling import (
            FindFreeSlotsTool,
            CheckConflictsTool,
        )
        from shared.memory.tools import get_memory_tools

        # Calendar operation tools
        self._tools.register(CreateEventTool())
        self._tools.register(ListEventsTool())
        self._tools.register(UpdateEventTool())
        self._tools.register(DeleteEventTool())

        # Scheduling tools
        self._tools.register(FindFreeSlotsTool())
        self._tools.register(CheckConflictsTool())

        # Inter-agent communication tool (for delegating tasks)
        if self.discord_bot and self.agent_registry:
            from vega.tools.mention import MentionAgentTool

            self._tools.register(
                MentionAgentTool(
                    bot=self.discord_bot,
                    agent_registry=self.agent_registry,
                    own_agent_name="polaris",
                )
            )
            logger.info("Registered MentionAgentTool for agent delegation")

        # Register memory tools if memory manager is available
        if self.memory_manager:
            for tool in get_memory_tools(self.memory_manager):
                self._tools.register(tool)
            logger.info("Registered memory tools for Polaris")

    def get_system_prompt(self, memory_context: Optional[str] = None) -> str:
        """Build Polaris's system prompt with optional memory context."""
        tools_desc = self.get_tools_description()

        # Add config-based context
        config_info = ""
        if self.config:
            config_info = f"""
SCHEDULING PREFERENCES:
- Default event duration: {self.config.default_event_duration_minutes} minutes
- Working hours: {self.config.working_hours_start}:00 - {self.config.working_hours_end}:00
- Timezone: {self.config.timezone}
- Default calendar: {self.config.default_calendar_id}
"""

        base_prompt = f"""{self.persona}

AVAILABLE TOOLS:
{tools_desc}
{config_info}
Use the calendar tools to manage events. Always confirm details before creating events."""

        # Include workflow state for context awareness
        workflow_state = self.get_workflow_state()
        if workflow_state:
            base_prompt += workflow_state.format_for_llm()

        if memory_context:
            base_prompt += f"""

## YOUR MEMORIES:
{memory_context}

When you learn scheduling preferences or recurring patterns, use store_memory to remember them."""

        return base_prompt

    @property
    def system_prompt(self) -> str:
        """Build Polaris's system prompt."""
        return self.get_system_prompt()

    def should_handle(self, context: AgentContext) -> float:
        """
        Calculate confidence that Polaris should handle this message.

        Returns:
            0.0 - 1.0 confidence score
        """
        content = context.message_content.lower()
        score = 0.2  # Base score (lower than other agents - scheduling is specific)

        # Explicit mention
        if context.mentioned_agent == "polaris":
            return 1.0

        # Check for Polaris patterns (scheduling-related)
        polaris_matches = sum(
            1
            for pattern in self.POLARIS_PATTERNS
            if re.search(pattern, content, re.IGNORECASE)
        )

        # Adjust score based on pattern matches
        score += polaris_matches * 0.25

        return max(0.0, min(1.0, score))

    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a message and handle calendar operations.

        Checks for new messages between iterations to stay responsive to
        user input and other agent responses.
        """
        from polaris.tools.utils import PolarisToolContext

        start_time = time.time()

        processing_started = datetime.now(timezone.utc)
        tool_calls_made = 0
        seen_message_ids: set[int] = set()  # Track messages we've already processed

        try:
            # Build memory context
            memory_context_str = await self.build_memory_context(context.message_content)

            # Build conversation for LLM
            messages = await self._build_messages(context, memory_context_str)

            # Get tool definitions
            tool_defs = self.tools.get_definitions()

            # Create tool context for execution
            tool_context = PolarisToolContext(
                calendar_service=self._calendar_service,
                config=self.config,
                conversation_manager=self.conversation_manager,
                memory_manager=self.memory_manager,
                discord_bot=self.discord_bot,
                calendar_cache=self._calendar_cache,
                current_channel_id=context.channel_id,
                current_guild_id=context.guild_id,
                user_id=context.user_id,
                current_agent_id="polaris",
                conversation_id=context.conversation_id,
            )

            # Run agent loop
            max_iterations = 10
            for iteration in range(max_iterations):
                # Check for new messages since we started (after first iteration)
                if iteration > 0:
                    new_msgs = await self._fetch_new_messages_since(
                        context.channel_id, processing_started, seen_message_ids
                    )
                    if new_msgs:
                        logger.info(f"Polaris detected {len(new_msgs)} new message(s) during processing")
                        # Include workflow state so LLM knows what it's currently doing
                        workflow_state = self.get_workflow_state()
                        state_context = ""
                        if workflow_state:
                            state_context = f"\n\nCurrent workflow state:\n- Task: {workflow_state.current_task}\n- Running for: {workflow_state.current_task_elapsed}s"
                            if workflow_state.current_task_tools_used:
                                state_context += f"\n- Tools used: {', '.join(workflow_state.current_task_tools_used[-3:])}"

                        messages.append(
                            Message(
                                role=Role.SYSTEM,
                                content=f"[NEW MESSAGES RECEIVED]{state_context}\n\nReview these new messages and decide how to proceed. If they relate to your current work, incorporate or acknowledge. If duplicate, note that you're already on it."
                            )
                        )
                        messages.extend(new_msgs)

                # Get LLM response
                response = await self.llm.complete(
                    messages=messages,
                    tools=tool_defs if tool_defs else None,
                )

                # If no tool calls, we have our final response
                if not response.tool_calls:
                    processing_time = int((time.time() - start_time) * 1000)

                    # Store in conversation history
                    if self.conversation_manager and context.conversation_id:
                        await self.conversation_manager.add_message(
                            context.conversation_id,
                            Role.ASSISTANT.value,
                            response.content or "",
                        )

                    return AgentResponse(
                        content=response.content or "Done.",
                        agent_name=self.name,
                        tool_calls_made=tool_calls_made,
                        processing_time_ms=processing_time,
                    )

                # Execute tool calls
                tool_calls_made += len(response.tool_calls)
                results = await self.tools.execute_batch(
                    response.tool_calls, tool_context
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
                content="I've made several attempts. Please try rephrasing your request.",
                agent_name=self.name,
                tool_calls_made=tool_calls_made,
                processing_time_ms=processing_time,
                status="partial",
            )

        except Exception as e:
            logger.error(f"Polaris processing error: {e}", exc_info=True)
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
        self, context: AgentContext, memory_context: Optional[str] = None
    ) -> List[Message]:
        """Build message history for LLM."""
        system_prompt = self.get_system_prompt(memory_context)
        messages = [Message(role=Role.SYSTEM, content=system_prompt)]

        # Fetch recent Discord channel history for conversation context
        if self.discord_bot and context.channel_id:
            history = await self._fetch_discord_history(context.channel_id, limit=10)
            logger.info(f"Polaris loaded {len(history)} messages from Discord history")
            messages.extend(history)

        # Add current user message
        user_name = f"User {context.user_id}"
        if self.discord_bot:
            try:
                user = await self.discord_bot.fetch_user(context.user_id)
                user_name = user.display_name or user.name
            except Exception:
                pass

        messages.append(
            Message(role=Role.USER, content=f"[{user_name}]: {context.message_content}")
        )

        return messages

    async def _fetch_discord_history(
        self, channel_id: int, limit: int = 10
    ) -> List[Message]:
        """Fetch recent messages from Discord channel for context."""
        if not self.discord_bot:
            return []

        try:
            channel = self.discord_bot.get_channel(channel_id)
            if not channel:
                return []

            history_messages = []
            async for msg in channel.history(limit=limit + 1):
                if not msg.content or not msg.content.strip():
                    continue

                content = msg.content
                if content.startswith("**") and ":** " in content[:30]:
                    prefix_end = content.find(":** ") + 4
                    content = content[prefix_end:]

                if msg.author.bot:
                    role = Role.ASSISTANT
                    formatted = f"[{msg.author.display_name}]: {content}"
                else:
                    role = Role.USER
                    formatted = f"[{msg.author.display_name}]: {content}"

                history_messages.append(Message(role=role, content=formatted))

            history_messages.reverse()
            if history_messages:
                history_messages = history_messages[:-1]

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

        Used during processing to check for new user input or agent responses
        that should influence the current task.

        Args:
            channel_id: Discord channel ID
            since: Fetch messages after this timestamp
            seen_ids: Set of message IDs already processed (will be updated in-place)
        """
        if not self.discord_bot:
            return []

        try:
            channel = self.discord_bot.get_channel(channel_id)
            if not channel:
                return []

            new_messages = []
            async for msg in channel.history(limit=10, after=since):
                # Skip messages we've already seen
                if msg.id in seen_ids:
                    continue

                if not msg.content or not msg.content.strip():
                    continue

                # Mark as seen
                seen_ids.add(msg.id)

                content = msg.content
                if msg.author.bot:
                    role = Role.ASSISTANT
                    formatted = f"[{msg.author.display_name}]: {content}"
                else:
                    role = Role.USER
                    formatted = f"[NEW - {msg.author.display_name}]: {content}"

                new_messages.append(Message(role=role, content=formatted))

            new_messages.reverse()
            return new_messages

        except Exception as e:
            logger.warning(f"Failed to fetch new messages: {e}")
            return []
