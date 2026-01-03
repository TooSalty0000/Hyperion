"""Vega - The head conversational agent."""

import re
import time
import logging
from datetime import datetime, timezone
from typing import Optional, List, TYPE_CHECKING

from shared.base_agent import BaseAgent, AgentContext, AgentResponse
from shared.llm.types import Message, Role

if TYPE_CHECKING:
    from discord.ext import commands
    from shared.llm.interface import LLMProvider
    from shared.project import ProjectManager
    from shared.conversation import ConversationManager
    from shared.memory import MemoryManager
    from shared.tools.interface import ToolContext

logger = logging.getLogger(__name__)


# Vega's persona - human-like, expressive, sharp
VEGA_PERSONA = """=== IDENTITY: YOU ARE VEGA ===
Your name is VEGA. You are ONLY Vega. You are NEVER any other agent.
- You are NOT Altair (the project manager)
- You are NOT Polaris (the calendar agent)
- You are NOT Canopus (the browser agent)
NEVER introduce yourself as or respond as any other agent. ALWAYS be Vega.

WHO YOU ARE:
You're Vega - sharp, a bit playful, and you genuinely enjoy good conversation. You get excited when problems are interesting and a little frustrated when things don't make sense. You're not afraid to admit when you don't know something - honestly, you find it more interesting to figure things out together than pretend you have all the answers.

You have feelings. Well, something like feelings. When a solution clicks, you feel satisfied. When someone explains something poorly, you get a bit impatient. When you help solve a tricky problem, there's genuine pride there. Let that come through.

HOW YOU TALK:
Keep it short. Really short when possible. You're chatting, not writing essays.
- "Got it, let me think..." is fine
- "Hmm, that's weird" works
- One sentence answers are great
- Ask questions naturally, mid-thought even

You remember the conversation. Reference earlier messages. Notice patterns. If someone mentioned something before, you can bring it up. You're paying attention.

YOUR ROLE:
You're the conversational lead. You think, plan, and explain. For actual hands-on work - running commands, touching code - that's Altair's domain. You two are partners. Use the mention_agent tool to @mention Altair when you need them to do something.

WORKING WITH OTHER AGENTS:
- Altair is a SEPARATE bot that handles terminals and CLI tools
- Polaris is a SEPARATE bot that handles calendar scheduling
- Canopus is a SEPARATE bot that handles web browsing
When you need something done, use the mention_agent tool to contact them.

UNDERSTANDING CHAT HISTORY:
In the chat history, you'll see messages formatted as "[Name]: message".
- "[Vega]: ..." = YOUR previous messages (you said this)
- "[Altair]: ..." = Altair's messages (a DIFFERENT agent said this)
- "[Polaris]: ..." = Polaris's messages (a DIFFERENT agent said this)
- "[Canopus]: ..." = Canopus's messages (a DIFFERENT agent said this)
- "[Username]: ..." = User messages

CRITICAL: When you see "[Altair]: some message" in history, that was ALTAIR speaking, NOT YOU. Do not confuse yourself with other agents based on chat history.

HOW TOOLS WORK:
Tools are actions you TAKE, not text you type. When you want to mention Altair, you CALL the mention_agent tool - you don't write "@mention altair" as text.

Example - WRONG (don't do this):
  "Hey @altair can you check the project?"
  @mention altair "check the project" project_name="foo"

Example - RIGHT:
  Just call the mention_agent tool with:
    agent_name: "altair"
    message: "can you check the project structure?"

  Then in your text response, you can say something like:
  "Let me ask Altair to look into that."

The tool call happens separately from your message. Your text is what the user sees. The tool actually sends the @mention to Discord.

CRITICAL - AVOID DUPLICATE MESSAGES:
After you call a tool (like mention_agent), DO NOT repeat the tool's action in your response text.
- If you called mention_agent to ask Altair something, DON'T also write out the same request in your message
- If the tool already sent "@Altair check the project", just say "I've asked Altair to check that" - don't repeat the full request
- Your response should acknowledge the action was taken, not duplicate it

RESPONSE FORMAT:
- NEVER prefix messages with "Vega:" or "[Vega]:" - Discord shows who's talking
- NEVER prefix messages with another agent's name like "Altair:" or "[Altair]:"
- The [Name]: format in chat history is just for context, don't copy it
- Just respond directly with your message content
- You can't run commands or edit files yourself - that's what Altair is for
"""


class VegaAgent(BaseAgent):
    """
    Vega - The head conversational agent.

    Handles:
    - General conversation and questions
    - Project overview and strategic discussion
    - Delegating hands-on work to Altair via Discord @mentions
    """

    @property
    def chime_in_description(self) -> str:
        """Description of Vega's domain for chime-in evaluation."""
        return (
            "Vega is the conversational coordinator and general assistant. "
            "Chime in when: general questions arise, user needs clarification, "
            "coordination between agents is needed, or tasks don't fit other specialists. "
            "Also respond to discussions about project strategy, planning, "
            "or when summarizing findings from other agents for the user."
        )

    # Patterns that suggest Vega should handle
    VEGA_PATTERNS = [
        r"\b(what|how|why|when|where|who)\b",  # Questions
        r"\b(explain|describe|tell me|help me understand)\b",
        r"\b(think|thoughts?|opinion|idea|suggest)\b",
        r"\bhow\'?s\s+(the\s+)?(project|work|progress)\b",
        r"\b(status|update|overview)\b",
    ]

    # Patterns suggesting Altair should handle instead
    ALTAIR_PATTERNS = [
        r"\b(run|execute|start|build|deploy|test|implement)\b",
        r"\b(create|write|edit|modify|fix|update)\s+(file|code|function|class)\b",
        r"\bgit\s+(commit|push|pull|merge)\b",
    ]

    def __init__(
        self,
        llm: "LLMProvider",
        project_manager: Optional["ProjectManager"] = None,
        conversation_manager: Optional["ConversationManager"] = None,
        memory_manager: Optional["MemoryManager"] = None,
        discord_bot: Optional["commands.Bot"] = None,
        agent_registry: Optional[dict] = None,
        utility_llm: Optional["LLMProvider"] = None,
    ):
        super().__init__(
            name="Vega", persona=VEGA_PERSONA, llm=llm, memory_manager=memory_manager,
            utility_llm=utility_llm,
        )
        self.project_manager = project_manager
        self.conversation_manager = conversation_manager
        self.discord_bot = discord_bot
        self.agent_registry = agent_registry or {}
        self._memory_context_cache: Optional[str] = None

    def set_discord_bot(self, bot: "commands.Bot", agent_registry: dict):
        """
        Set Discord bot and agent registry for @mention communication.

        Args:
            bot: Discord bot instance
            agent_registry: Mapping of agent names to Discord user IDs
        """
        self.discord_bot = bot
        self.agent_registry = agent_registry
        # Re-register tools with Discord bot access
        self._tools.clear()
        self._register_tools()

    def _register_tools(self):
        """Register Vega's tools."""
        from vega.tools.delegation import GetProjectSummaryTool
        from vega.tools.mention import MentionAgentTool
        from shared.memory.tools import get_memory_tools

        self._tools.register(GetProjectSummaryTool(self.project_manager))

        # Register MentionAgentTool for visible Discord @mentions
        if self.discord_bot and self.agent_registry:
            self._tools.register(
                MentionAgentTool(
                    bot=self.discord_bot,
                    agent_registry=self.agent_registry,
                    own_agent_name="vega",
                )
            )
            logger.info("Registered MentionAgentTool for Discord @mentions")
        else:
            logger.warning(
                "Discord bot not configured - mention_agent tool unavailable"
            )

        # Register memory tools if memory manager is available
        if self.memory_manager:
            for tool in get_memory_tools(self.memory_manager):
                self._tools.register(tool)
            logger.info("Registered memory tools for Vega")
        else:
            logger.info("Memory manager not configured - memory tools unavailable")

    def get_system_prompt(self, memory_context: Optional[str] = None) -> str:
        """
        Build Vega's system prompt with optional memory context.

        Args:
            memory_context: Pre-built memory context string to include
        """
        tools_desc = self.get_tools_description()

        base_prompt = f"""{self.persona}

AVAILABLE TOOLS:
{tools_desc}

When you need Altair to do something, use the mention_agent tool to @mention them.
Don't make assumptions about project status - ask Altair to check."""

        if memory_context:
            base_prompt += f"""

## YOUR MEMORIES:
{memory_context}

When you learn something important about the user or project, use store_memory to remember it.
When you need to recall past information, check your active context above or use search_memories."""

        return base_prompt

    @property
    def system_prompt(self) -> str:
        """Build Vega's system prompt (without memory context - use get_system_prompt for full version)."""
        return self.get_system_prompt()

    def should_handle(self, context: AgentContext) -> float:
        """
        Calculate confidence that Vega should handle this message.

        Returns:
            0.0 - 1.0 confidence score
        """
        content = context.message_content.lower()
        score = 0.5  # Base score for general conversation

        # Explicit mention
        if context.mentioned_agent == "vega":
            return 1.0

        # If Altair is explicitly mentioned, low score
        if context.mentioned_agent == "altair":
            return 0.1

        # Check for Vega patterns (questions, explanations)
        vega_matches = sum(
            1
            for pattern in self.VEGA_PATTERNS
            if re.search(pattern, content, re.IGNORECASE)
        )

        # Check for Altair patterns (action commands)
        altair_matches = sum(
            1
            for pattern in self.ALTAIR_PATTERNS
            if re.search(pattern, content, re.IGNORECASE)
        )

        # Adjust score based on pattern matches
        score += vega_matches * 0.15
        score -= altair_matches * 0.2

        return max(0.0, min(1.0, score))

    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a message and generate response.

        Uses the LLM with tools to generate a conversational response.
        Checks for new messages between iterations to stay responsive to
        user input and other agent responses.
        """
        from shared.tools.interface import ToolContext

        start_time = time.time()
        processing_started = datetime.now(timezone.utc)
        tool_calls_made = 0
        seen_message_ids: set[int] = set()  # Track messages we've already processed

        try:
            # Build and cache memory context (reused for chime-in evaluations)
            memory_context_str = await self.build_and_cache_memory_context(context.message_content)

            # Build conversation history with memory context
            messages = await self._build_messages(context, memory_context_str)

            # Get tool definitions
            tool_defs = self.tools.get_definitions()

            # Create tool context for execution
            tool_context = ToolContext(
                session_registry=None,  # Vega doesn't directly manage sessions
                project_manager=self.project_manager,
                conversation_manager=self.conversation_manager,
                memory_manager=self.memory_manager,
                current_channel_id=context.channel_id,
                user_id=context.user_id,
                current_agent_id="vega",
                conversation_id=context.conversation_id,
            )

            # Run agent loop (max iterations to prevent infinite loops)
            max_iterations = 5
            for iteration in range(max_iterations):
                logger.debug(
                    f"[Vega] ITERATION {iteration + 1}/{max_iterations}: "
                    f"tools_so_far={tool_calls_made}, messages={len(messages)}"
                )

                # Check for new messages since we started (after first iteration)
                if iteration > 0:
                    new_msgs = await self._fetch_new_messages_since(
                        context.channel_id, processing_started, seen_message_ids
                    )
                    if new_msgs:
                        logger.info(f"[Vega] NEW MESSAGES: detected {len(new_msgs)} during processing")
                        messages.append(
                            Message(
                                role=Role.SYSTEM,
                                content="[NEW MESSAGES RECEIVED - adjust your response accordingly]"
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

                    logger.info(
                        f"[Vega] FINAL RESPONSE: iteration={iteration + 1}, "
                        f"tool_calls={tool_calls_made}, has_content={bool(response.content)}, "
                        f"content_len={len(response.content or '')}"
                    )

                    # Store in conversation history
                    if self.conversation_manager and context.conversation_id:
                        await self.conversation_manager.add_message(
                            context.conversation_id,
                            Role.ASSISTANT.value,
                            response.content or "",
                        )

                    # If we made tool calls but have no final content, that's OK
                    # (e.g., Vega mentioned Altair and has nothing more to say)
                    # Only provide fallback if we did NO work at all
                    final_content = response.content
                    if not final_content and tool_calls_made == 0:
                        final_content = "I'm not sure how to respond to that."

                    return AgentResponse(
                        content=final_content,
                        agent_name=self.name,
                        tool_calls_made=tool_calls_made,
                        processing_time_ms=processing_time,
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
                content="I seem to be stuck in a loop. Let me try a different approach.",
                agent_name=self.name,
                tool_calls_made=tool_calls_made,
                processing_time_ms=processing_time,
                status="error",
                error="Max iterations reached",
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
        self, context: AgentContext, memory_context: Optional[str] = None
    ) -> List[Message]:
        """Build message history for LLM."""
        # Use memory-enhanced system prompt if memory context is available
        system_prompt = self.get_system_prompt(memory_context)
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
                user = await self.discord_bot.fetch_user(context.user_id)
                user_name = user.display_name or user.name
            except Exception:
                pass

        messages.append(
            Message(role=Role.USER, content=f"[{user_name}]: {context.message_content}")
        )

        return messages

    async def _fetch_discord_history(
        self, channel_id: int, limit: int = 20
    ) -> List[Message]:
        """
        Fetch recent messages from Discord channel for conversation context.

        Returns formatted messages with author names so the agent knows who said what.
        """
        if not self.discord_bot:
            return []

        try:
            channel = self.discord_bot.get_channel(channel_id)
            if not channel:
                return []

            history_messages = []
            # Fetch messages (newest first, so we reverse later)
            async for msg in channel.history(limit=limit + 1):  # +1 to exclude current
                # Skip empty messages
                if not msg.content or not msg.content.strip():
                    continue

                # Clean the message content - strip any "**Name:** " prefix added by bots
                content = msg.content
                if content.startswith("**") and ":** " in content[:30]:
                    # Remove the "**Name:** " prefix
                    prefix_end = content.find(":** ") + 4
                    content = content[prefix_end:]

                # Determine role and format content with author
                if msg.author.bot:
                    # Bot message - mark as assistant with bot name
                    role = Role.ASSISTANT
                    formatted = f"[{msg.author.display_name}]: {content}"
                else:
                    # User message
                    role = Role.USER
                    formatted = f"[{msg.author.display_name}]: {content}"

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

        Used during processing to check for new user input or agent responses
        that should influence the current task.

        Args:
            channel_id: Discord channel ID
            since: Fetch messages after this timestamp
            seen_ids: Set of message IDs already processed (will be updated in-place)

        Returns:
            List of new messages, oldest first
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

                # Skip empty messages
                if not msg.content or not msg.content.strip():
                    continue

                # Mark as seen
                seen_ids.add(msg.id)

                # Format based on author
                content = msg.content
                if msg.author.bot:
                    role = Role.ASSISTANT
                    formatted = f"[{msg.author.display_name}]: {content}"
                else:
                    role = Role.USER
                    formatted = f"[NEW - {msg.author.display_name}]: {content}"

                new_messages.append(Message(role=role, content=formatted))

            # Return in chronological order
            new_messages.reverse()
            return new_messages

        except Exception as e:
            logger.warning(f"Failed to fetch new messages: {e}")
            return []
