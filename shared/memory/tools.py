"""Memory tools for agent persistent memory."""

import logging
from typing import Optional, List, TYPE_CHECKING

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

if TYPE_CHECKING:
    from .manager import MemoryManager

logger = logging.getLogger(__name__)


class StoreMemoryTool(Tool):
    """
    Store a new memory.

    Allows agents to remember important information across conversations.
    """

    def __init__(self, memory_manager: Optional['MemoryManager'] = None):
        self._memory_manager = memory_manager

    @property
    def name(self) -> str:
        return "store_memory"

    @property
    def description(self) -> str:
        return (
            "Store something important to remember for future conversations. "
            "Assign an importance score (0.0-1.0) based on how valuable this information is. "
            "High importance (>0.8) items stay readily accessible in your active context. "
            "Lower importance items may be summarized and categorized for later retrieval. "
            "Use this when you learn user preferences, project details, or important context."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="content",
                type="string",
                description="The information to remember. Be specific and include relevant context.",
                required=True,
            ),
            ToolParameter(
                name="importance",
                type="number",
                description=(
                    "Importance score 0.0-1.0. "
                    "0.9-1.0: Critical (user identity, core preferences). "
                    "0.7-0.8: Important (project details, recurring patterns). "
                    "0.5-0.6: Useful (temporary context, minor preferences). "
                    "0.3-0.4: Low (background info that might be useful later)."
                ),
                required=True,
            ),
            ToolParameter(
                name="category",
                type="string",
                description=(
                    "Optional category: user_preferences, project_context, technical_notes, "
                    "workflow_patterns, or conversation_highlights. "
                    "Leave empty for auto-categorization."
                ),
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        content: str,
        importance: float,
        category: str = None,
    ) -> str:
        """Store a new memory."""
        memory_manager = self._memory_manager or getattr(context, 'memory_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        if not memory_manager:
            return "Memory system is not available. Configure REDIS_URL to enable."

        if not agent_id:
            return "Error: Agent ID not set in context."

        try:
            # Clamp importance to valid range
            importance = max(0.0, min(1.0, float(importance)))

            memory = await memory_manager.store_memory(
                agent_id=agent_id,
                content=content,
                importance=importance,
                category=category,
                source_conversation_id=getattr(context, 'conversation_id', None),
            )

            importance_level = (
                "critical" if importance >= 0.9 else
                "important" if importance >= 0.7 else
                "useful" if importance >= 0.5 else
                "background"
            )

            return (
                f"Stored memory (ID: {memory.id[:8]}...)\n"
                f"Importance: {importance:.1f} ({importance_level})\n"
                f"Keywords: {', '.join(memory.keywords[:5])}\n"
                f"This memory is now in your active context."
            )

        except Exception as e:
            logger.error(f"Error storing memory: {e}", exc_info=True)
            return f"Error storing memory: {str(e)}"


class SearchMemoriesTool(Tool):
    """
    Search and retrieve memories.

    Allows agents to find relevant information from long-term memory.
    """

    def __init__(self, memory_manager: Optional['MemoryManager'] = None):
        self._memory_manager = memory_manager

    @property
    def name(self) -> str:
        return "search_memories"

    @property
    def description(self) -> str:
        return (
            "Search your long-term memories for relevant information. "
            "Use when you need to recall details not in your active context. "
            "Can filter by category or search across all memories. "
            "Your active context already includes important short-term memories - "
            "this tool searches the categorized long-term archive."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="What to search for. Use specific keywords for better results.",
                required=True,
            ),
            ToolParameter(
                name="category",
                type="string",
                description=(
                    "Optional category to search within: user_preferences, project_context, "
                    "technical_notes, workflow_patterns, conversation_highlights. "
                    "Leave empty to search all categories."
                ),
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum results to return (default 5, max 10).",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        query: str,
        category: str = None,
        limit: int = 5,
    ) -> str:
        """Search memories."""
        memory_manager = self._memory_manager or getattr(context, 'memory_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        if not memory_manager:
            return "Memory system is not available. Configure REDIS_URL to enable."

        if not agent_id:
            return "Error: Agent ID not set in context."

        try:
            limit = min(max(1, int(limit)), 10)  # Clamp to 1-10

            memories = await memory_manager.search_memories(
                agent_id=agent_id,
                query=query,
                category=category,
                limit=limit,
            )

            if not memories:
                if category:
                    return f"No memories found matching '{query}' in category '{category}'."
                return f"No memories found matching '{query}'."

            lines = [f"Found {len(memories)} matching memories:"]
            for i, memory in enumerate(memories, 1):
                cat_str = f"[{memory.category}] " if memory.category else ""
                content = memory.summary or memory.content
                if len(content) > 100:
                    content = content[:100] + "..."
                lines.append(f"{i}. {cat_str}{content}")
                lines.append(f"   (importance: {memory.importance:.1f}, id: {memory.id[:8]})")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error searching memories: {e}", exc_info=True)
            return f"Error searching memories: {str(e)}"


class PromoteMemoryTool(Tool):
    """
    Promote a memory to long-term storage with explicit category.

    Allows agents to archive important context while freeing active space.
    """

    def __init__(self, memory_manager: Optional['MemoryManager'] = None):
        self._memory_manager = memory_manager

    @property
    def name(self) -> str:
        return "promote_memory"

    @property
    def description(self) -> str:
        return (
            "Move a short-term memory to long-term storage with a specific category. "
            "Use when you want to archive important information for future reference "
            "while freeing up active context space. The memory will be summarized "
            "and indexed for retrieval via search_memories."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="memory_id",
                type="string",
                description="ID of the memory to promote (from your active context).",
                required=True,
            ),
            ToolParameter(
                name="category",
                type="string",
                description=(
                    "Category for the memory: user_preferences, project_context, "
                    "technical_notes, workflow_patterns, or conversation_highlights."
                ),
                required=True,
            ),
            ToolParameter(
                name="summary",
                type="string",
                description="Optional custom summary for the memory. If not provided, one is auto-generated.",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        memory_id: str,
        category: str,
        summary: str = None,
    ) -> str:
        """Promote memory to long-term storage."""
        memory_manager = self._memory_manager or getattr(context, 'memory_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        if not memory_manager:
            return "Memory system is not available. Configure REDIS_URL to enable."

        if not agent_id:
            return "Error: Agent ID not set in context."

        try:
            success = await memory_manager.promote_to_long_term(
                agent_id=agent_id,
                memory_id=memory_id,
                category=category,
                summary=summary,
            )

            if not success:
                return f"Memory '{memory_id}' not found or already in long-term storage."

            return (
                f"Memory promoted to long-term storage.\n"
                f"Category: {category}\n"
                f"You can retrieve it later with: search_memories(query=\"...\", category=\"{category}\")"
            )

        except Exception as e:
            logger.error(f"Error promoting memory: {e}", exc_info=True)
            return f"Error promoting memory: {str(e)}"


class ForgetMemoryTool(Tool):
    """
    Delete a memory that is no longer relevant or accurate.
    """

    def __init__(self, memory_manager: Optional['MemoryManager'] = None):
        self._memory_manager = memory_manager

    @property
    def name(self) -> str:
        return "forget_memory"

    @property
    def description(self) -> str:
        return (
            "Remove a memory that is no longer relevant or accurate. "
            "Use sparingly - only when information is outdated, incorrect, "
            "or explicitly requested to be forgotten by the user."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="memory_id",
                type="string",
                description="ID of the memory to delete.",
                required=True,
            ),
            ToolParameter(
                name="reason",
                type="string",
                description="Brief reason for deleting this memory (for logging).",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        memory_id: str,
        reason: str = None,
    ) -> str:
        """Delete a memory."""
        memory_manager = self._memory_manager or getattr(context, 'memory_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        if not memory_manager:
            return "Memory system is not available. Configure REDIS_URL to enable."

        if not agent_id:
            return "Error: Agent ID not set in context."

        try:
            # Get memory first to log what's being deleted
            memory = await memory_manager.get_memory(agent_id, memory_id)

            success = await memory_manager.delete_memory(
                agent_id=agent_id,
                memory_id=memory_id,
            )

            if not success:
                return f"Memory '{memory_id}' not found."

            reason_str = f" Reason: {reason}" if reason else ""
            logger.info(f"Memory {memory_id} deleted by {agent_id}.{reason_str}")

            content_preview = memory.content[:50] + "..." if memory and len(memory.content) > 50 else (memory.content if memory else "unknown")
            return f"Memory deleted: \"{content_preview}\""

        except Exception as e:
            logger.error(f"Error deleting memory: {e}", exc_info=True)
            return f"Error deleting memory: {str(e)}"


class ListMemoryCategoriesTool(Tool):
    """
    List available memory categories and their contents.
    """

    def __init__(self, memory_manager: Optional['MemoryManager'] = None):
        self._memory_manager = memory_manager

    @property
    def name(self) -> str:
        return "list_memory_categories"

    @property
    def description(self) -> str:
        return (
            "List all your long-term memory categories and their memory counts. "
            "Use this to see what organized knowledge you have stored."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        """List memory categories."""
        memory_manager = self._memory_manager or getattr(context, 'memory_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        if not memory_manager:
            return "Memory system is not available. Configure REDIS_URL to enable."

        if not agent_id:
            return "Error: Agent ID not set in context."

        try:
            categories = await memory_manager.get_category_summaries(agent_id)

            if not categories:
                return "No memory categories yet. Memories are categorized when moved to long-term storage."

            lines = ["Your memory categories:"]
            total_memories = 0
            for cat in categories:
                lines.append(f"- **{cat['name']}** ({cat['count']} memories)")
                lines.append(f"  {cat['description']}")
                total_memories += cat['count']

            lines.append(f"\nTotal long-term memories: {total_memories}")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error listing categories: {e}", exc_info=True)
            return f"Error listing categories: {str(e)}"


class BrowseMemoryCategoryTool(Tool):
    """
    Browse memories in a specific category.
    """

    def __init__(self, memory_manager: Optional['MemoryManager'] = None):
        self._memory_manager = memory_manager

    @property
    def name(self) -> str:
        return "browse_memory_category"

    @property
    def description(self) -> str:
        return (
            "Browse all memories in a specific category. "
            "Use this to review what you know about a topic area."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="category",
                type="string",
                description=(
                    "Category to browse: user_preferences, project_context, "
                    "technical_notes, workflow_patterns, or conversation_highlights."
                ),
                required=True,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum memories to return (default 10).",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        category: str,
        limit: int = 10,
    ) -> str:
        """Browse memories in a category."""
        memory_manager = self._memory_manager or getattr(context, 'memory_manager', None)
        agent_id = getattr(context, 'current_agent_id', None)

        if not memory_manager:
            return "Memory system is not available. Configure REDIS_URL to enable."

        if not agent_id:
            return "Error: Agent ID not set in context."

        try:
            limit = min(max(1, int(limit)), 20)  # Clamp to 1-20

            memories = await memory_manager.get_memories_by_category(
                agent_id=agent_id,
                category=category,
                limit=limit,
            )

            if not memories:
                return f"No memories in category '{category}'."

            lines = [f"Memories in '{category}' ({len(memories)} shown):"]
            for i, memory in enumerate(memories, 1):
                content = memory.summary or memory.content
                if len(content) > 120:
                    content = content[:120] + "..."
                lines.append(f"{i}. [{memory.importance:.1f}] {content}")
                lines.append(f"   (id: {memory.id[:8]}, accessed: {memory.access_count}x)")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error browsing category: {e}", exc_info=True)
            return f"Error browsing category: {str(e)}"


# Convenience function to get all memory tools
def get_memory_tools(memory_manager: Optional['MemoryManager'] = None) -> List[Tool]:
    """Get all memory tools with the given manager."""
    return [
        StoreMemoryTool(memory_manager),
        SearchMemoriesTool(memory_manager),
        PromoteMemoryTool(memory_manager),
        ForgetMemoryTool(memory_manager),
        ListMemoryCategoriesTool(memory_manager),
        BrowseMemoryCategoryTool(memory_manager),
    ]
