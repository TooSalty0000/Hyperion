"""Base classes for the multi-agent system."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.llm.interface import LLMProvider
    from shared.tools.registry import ToolRegistry
    from shared.collaboration.evaluator import ChimeInContext, ChimeInResult

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for agent execution."""
    channel_id: int
    user_id: int
    message_content: str
    guild_id: Optional[int] = None
    mentioned_agent: Optional[str] = None  # "vega", "altair", or None
    conversation_id: Optional[str] = None
    # Additional context that can be populated
    project_name: Optional[str] = None
    session_id: Optional[int] = None
    # Chime-in mode: agent decided to join conversation, keep response simple
    is_chime_in: bool = False


@dataclass
class AgentResponse:
    """Unified response from any agent."""
    content: str
    agent_name: str
    tool_calls_made: int = 0
    processing_time_ms: int = 0
    # Delegation support
    delegate_to: Optional[str] = None
    delegation_context: Optional[Dict[str, Any]] = None
    # Error handling
    error: Optional[str] = None
    status: str = "complete"  # "complete", "error", "delegated"


class BaseAgent(ABC):
    """
    Abstract base for all agents in the system.

    Each agent has:
    - A name and persona
    - Access to an LLM provider
    - A set of tools it can use
    - Optional memory manager for persistent memory
    - Logic to determine if it should handle a message
    """

    def __init__(
        self,
        name: str,
        persona: str,
        llm: 'LLMProvider',
        memory_manager: Optional[Any] = None,
        utility_llm: Optional['LLMProvider'] = None,
    ):
        self.name = name
        self.persona = persona
        self.llm = llm
        self.memory_manager = memory_manager  # Optional MemoryManager for persistent memory
        self.utility_llm = utility_llm  # Optional cheaper/faster LLM for quick evaluations
        self._tools: Optional['ToolRegistry'] = None

        # Cached memory context - updated during process(), reused for chime-in
        self._cached_memory_context: Optional[str] = None

    @property
    def tools(self) -> 'ToolRegistry':
        """Get the tool registry for this agent."""
        if self._tools is None:
            from shared.tools.registry import ToolRegistry
            self._tools = ToolRegistry()
            self._register_tools()
        return self._tools

    @abstractmethod
    def _register_tools(self):
        """Register tools available to this agent. Override in subclasses."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Agent's system prompt including persona and instructions."""
        pass

    @abstractmethod
    async def process(self, context: AgentContext) -> AgentResponse:
        """
        Process a message and return a response.

        This is the main entry point for agent execution.
        """
        pass

    @abstractmethod
    def should_handle(self, context: AgentContext) -> float:
        """
        Return confidence score (0.0-1.0) that this agent should handle the message.

        Used by the router to decide which agent gets the message.
        Higher score = more confident this agent should handle it.
        """
        pass

    def get_tools_description(self) -> str:
        """Get a formatted description of available tools for the system prompt."""
        tools = self.tools.list_all()
        if not tools:
            return "No tools available."

        lines = []
        for tool in tools:
            params = ", ".join(
                f"{p.name}: {p.type}" + (" (required)" if p.required else "")
                for p in tool.parameters
            )
            lines.append(f"- **{tool.name}**({params}): {tool.description}")

        return "\n".join(lines)

    async def build_and_cache_memory_context(self, message_content: str) -> Optional[str]:
        """
        Build memory context and cache it for reuse by chime-in evaluations.

        Call this during process() to both get the memory context string
        AND cache it for later chime-in evaluations (avoids duplicate Redis calls).

        Args:
            message_content: Current message for relevance-based memory selection

        Returns:
            Formatted memory context string, or None if no memory manager
        """
        if not self.memory_manager:
            return None

        try:
            memory_context = await self.memory_manager.build_memory_context(
                agent_id=self.name.lower(),
                current_message=message_content,
            )
            self._cached_memory_context = memory_context.format_for_prompt()
            return self._cached_memory_context
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to build memory context: {e}")
            return None

    def get_cached_memory_context(self) -> str:
        """
        Get the cached memory context for chime-in evaluations.

        Returns the memory context that was built during the last process() call.
        Falls back to empty string if no cache available.
        """
        return self._cached_memory_context or ""

    @property
    def chime_in_description(self) -> str:
        """
        Description of this agent's domain/personality for chime-in evaluation.

        Override in subclasses to provide a more specific description.
        Default returns the persona.
        """
        return self.persona

    async def evaluate_chime_in(
        self,
        context: 'ChimeInContext',
        memory_context: str = "",
    ) -> 'ChimeInResult':
        """
        Evaluate whether this agent should chime in on a conversation.

        This is called when a message appears in the main channel that
        this agent wasn't directly mentioned in. The agent decides
        whether it has something valuable to contribute.

        Args:
            context: ChimeInContext with message details
            memory_context: Relevant memories about chime-in preferences

        Returns:
            ChimeInResult with decision and optional response
        """
        from shared.collaboration.evaluator import (
            ChimeInDecision,
            ChimeInResult,
            CHIME_IN_DECISION_SCHEMA,
            build_evaluation_prompt,
            parse_chime_in_response,
        )

        # Build evaluation prompt
        prompt = build_evaluation_prompt(
            agent_name=self.name,
            agent_description=self.chime_in_description,
            context=context,
            memory_context=memory_context,
        )

        try:
            from shared.llm.types import Message, Role

            # Use utility LLM if available (cheaper/faster), otherwise fall back to main LLM
            eval_llm = self.utility_llm or self.llm
            response = await eval_llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                max_tokens=200,  # Keep response short
                response_schema=CHIME_IN_DECISION_SCHEMA,  # Structured output
            )

            # Log raw LLM response for debugging
            logger.info(f"[{self.name}] Chime-in LLM raw: {response.content[:300]}")

            result = parse_chime_in_response(response.content)

            logger.debug(
                f"[{self.name}] Chime-in evaluation: {result.decision.value} "
                f"(confidence={result.confidence:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"[{self.name}] Chime-in evaluation failed: {e}")
            # Default to staying quiet on error
            return ChimeInResult(
                decision=ChimeInDecision.STAY_QUIET,
                reasoning=f"Evaluation error: {e}",
                confidence=0.0,
            )
