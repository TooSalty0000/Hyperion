"""Tool interface and context for agent execution."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, TYPE_CHECKING

from shared.llm.types import ToolDefinition, ToolParameter

if TYPE_CHECKING:
    from shared.session import SessionRegistry
    from shared.channel import ChannelManager


class CancellationToken:
    """Cooperative cancellation for long-running tool operations.

    Tools can check ``is_cancelled`` or use ``sleep()`` as a drop-in
    replacement for ``asyncio.sleep()`` that raises ``CancelledError``
    when the token is signalled.
    """

    def __init__(self):
        self._event = asyncio.Event()

    def cancel(self):
        """Signal cancellation."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def sleep(self, seconds: float):
        """Cancellation-aware sleep. Raises ``CancelledError`` if cancelled."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=seconds)
            # event.wait() returned → cancellation was signalled
            raise asyncio.CancelledError("Task cancelled via CancellationToken")
        except asyncio.TimeoutError:
            pass  # Normal timeout, continue


@dataclass
class ToolContext:
    """
    Context passed to tool execution.

    Provides access to system components needed by tools.
    """
    session_registry: Any = None  # SessionRegistry - avoid circular import
    project_manager: Optional[Any] = None  # ProjectManager
    channel_manager: Optional[Any] = None  # ChannelManager
    conversation_manager: Optional[Any] = None  # ConversationManager
    memory_manager: Optional[Any] = None  # MemoryManager for agent memory
    discord_bot: Optional[Any] = None  # Discord bot instance for channel creation
    permission_manager: Optional[Any] = None  # PermissionManager for approval requests
    browser_manager: Optional[Any] = None  # BrowserSessionManager for Canopus
    scratchpad: Optional[Any] = None  # NavigationScratchpad for Canopus cognitive tools

    # Current context state
    current_session_id: Optional[int] = None
    current_channel_id: Optional[int] = None
    current_guild_id: Optional[int] = None  # For channel creation
    user_id: Optional[int] = None
    current_agent_id: Optional[str] = None  # Agent ID for memory isolation ("vega" or "altair")
    conversation_id: Optional[str] = None  # Current conversation ID

    # Callback for starting output loops (set by agent/cog)
    start_output_loop_callback: Optional[Any] = None

    # Job graph executor (for Vega's orchestration tools)
    job_executor: Optional[Any] = None  # JobGraphExecutor

    # Trigger message (for thread creation in graph tools)
    trigger_message: Optional[Any] = None  # discord.Message

    # Response message (set by RespondToUserTool, read by cog)
    response_message: Optional[str] = None

    # Cancellation token for cooperative interruption of long-running tools.
    cancellation_token: Optional['CancellationToken'] = None

    # True when this agent was dispatched by another agent (has a node marker).
    # Used by cognitive tools to skip redundant handoff messages.
    is_dispatched: bool = False

    # Set by graph tools when DISPATCH nodes are sent to agents.
    # Signals Vega's process loop to yield and wait for agent responses.
    dispatched_to_agents: bool = False

    # Override channel for response delivery (set by RespondToUserTool).
    # When set, the cog sends the final response to this channel
    # instead of the channel the processing context came from.
    response_channel_id: Optional[int] = None

    def with_session(self, session_id: int) -> 'ToolContext':
        """Create a new context with updated session ID."""
        return ToolContext(
            session_registry=self.session_registry,
            project_manager=self.project_manager,
            channel_manager=self.channel_manager,
            conversation_manager=self.conversation_manager,
            memory_manager=self.memory_manager,
            discord_bot=self.discord_bot,
            permission_manager=self.permission_manager,
            browser_manager=self.browser_manager,
            scratchpad=self.scratchpad,
            cancellation_token=self.cancellation_token,
            current_session_id=session_id,
            current_channel_id=self.current_channel_id,
            current_guild_id=self.current_guild_id,
            user_id=self.user_id,
            current_agent_id=self.current_agent_id,
            conversation_id=self.conversation_id,
            start_output_loop_callback=self.start_output_loop_callback,
            response_channel_id=self.response_channel_id,
        )


class Tool(ABC):
    """
    Base class for agent tools.

    Each tool represents an action the agent can take.
    Tools are registered with a ToolRegistry and invoked
    by the agent's execution loop when the LLM requests them.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Tool name (unique identifier).

        This name is used by the LLM to invoke the tool.
        Should be snake_case and descriptive.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description for the LLM.

        This description helps the LLM understand when
        and how to use the tool.
        """
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """
        List of parameters this tool accepts.

        Each parameter is converted to JSON schema for
        the LLM's function calling interface.
        """
        pass

    @abstractmethod
    async def execute(self, context: ToolContext, **kwargs) -> str:
        """
        Execute the tool with given arguments.

        Args:
            context: Execution context with registry access
            **kwargs: Tool-specific arguments matching parameters

        Returns:
            String result for the LLM to consume.
            Should be informative but concise.
        """
        pass

    def to_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for LLM."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters
        )
