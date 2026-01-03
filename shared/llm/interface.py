"""Abstract interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, AsyncIterator, Any

from .types import Message, LLMResponse, StreamChunk, ToolDefinition


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    Implementations should handle provider-specific:
    - Message format conversion
    - Tool/function calling format
    - Response parsing
    - Streaming
    """

    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize the provider.

        Args:
            api_key: API key for the provider
            model: Model identifier
            **kwargs: Provider-specific configuration
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (e.g., 'gemini', 'openai', 'anthropic')."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: Conversation history
            tools: Available tools for function calling
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific options

        Returns:
            LLMResponse with content and/or tool calls
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion from the LLM.

        Args:
            messages: Conversation history
            tools: Available tools for function calling
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific options

        Yields:
            StreamChunk objects as they arrive
        """
        pass

    @abstractmethod
    def format_messages(self, messages: List[Message]) -> Any:
        """
        Convert unified messages to provider-specific format.

        Args:
            messages: List of Message objects

        Returns:
            Provider-specific message format
        """
        pass

    @abstractmethod
    def format_tools(self, tools: List[ToolDefinition]) -> Any:
        """
        Convert unified tools to provider-specific format.

        Args:
            tools: List of ToolDefinition objects

        Returns:
            Provider-specific tool format
        """
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> LLMResponse:
        """
        Parse provider-specific response to unified format.

        Args:
            response: Provider-specific response object

        Returns:
            Unified LLMResponse
        """
        pass
