"""Tool for agents to consciously decide not to respond."""

import logging
from typing import List, Optional

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)

# Special marker returned when agent decides to stay quiet
STAY_QUIET_MARKER = "__STAY_QUIET__"


class StayQuietTool(Tool):
    """
    Tool for agents to consciously decide not to respond.

    This makes "doing nothing" an explicit action rather than
    a lack of action. It's used during chime-in evaluation
    when an agent determines they have nothing valuable to add.

    The tool can also check memory for user preferences about
    when to stay quiet (e.g., "don't interrupt my coding sessions").
    """

    @property
    def name(self) -> str:
        return "stay_quiet"

    @property
    def description(self) -> str:
        return (
            "Consciously decide not to respond to this message. "
            "Use this when you've evaluated the message and determined "
            "that you have nothing valuable to add, or when the user "
            "has previously asked you not to interrupt. "
            "This is a deliberate choice, not just silence."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="reason",
                type="string",
                description="Brief reason for staying quiet (for learning)",
                required=True,
            ),
            ToolParameter(
                name="store_preference",
                type="boolean",
                description="Whether to remember this as a preference for future similar situations",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        reason: str,
        store_preference: bool = False,
    ) -> str:
        """
        Execute the stay quiet decision.

        Returns a special marker that the agent loop recognizes
        as "don't send any message".
        """
        agent_id = context.current_agent_id or "unknown"

        logger.debug(
            f"[{agent_id}] Staying quiet: {reason}"
        )

        # Optionally store as a memory for future reference
        if store_preference and context.memory_manager:
            try:
                memory_content = f"Preference noted: Stay quiet when {reason}"
                await context.memory_manager.store_memory(
                    agent_id=agent_id,
                    content=memory_content,
                    memory_type="preference",
                    tags=["chime_in", "stay_quiet", "preference"],
                )
                logger.debug(f"[{agent_id}] Stored stay-quiet preference")
            except Exception as e:
                logger.warning(f"Failed to store preference: {e}")

        return STAY_QUIET_MARKER


async def check_quiet_preferences(
    context: ToolContext,
    message_content: str,
) -> Optional[str]:
    """
    Check memory for relevant stay-quiet preferences.

    Args:
        context: Tool context with memory manager
        message_content: The message being evaluated

    Returns:
        Memory context string if relevant preferences found,
        None otherwise.
    """
    if not context.memory_manager or not context.current_agent_id:
        return None

    try:
        # Search for relevant chime-in preferences
        memories = await context.memory_manager.search_memories(
            agent_id=context.current_agent_id,
            query=f"chime in preference quiet {message_content[:50]}",
            limit=3,
            tags=["chime_in", "preference"],
        )

        if memories:
            return "\n".join(
                f"- {m.content}" for m in memories
            )
    except Exception as e:
        logger.warning(f"Failed to check quiet preferences: {e}")

    return None
