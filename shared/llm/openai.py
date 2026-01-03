"""OpenAI API provider implementation (stub)."""

from typing import List, Optional, AsyncIterator, Any

from .interface import LLMProvider
from .types import Message, LLMResponse, StreamChunk, ToolDefinition


class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.

    This is a stub implementation. Full implementation requires:
    - pip install openai
    - OpenAI API key
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        **kwargs
    ):
        super().__init__(api_key, model, **kwargs)
        # Lazy import would go here
        raise NotImplementedError(
            "OpenAI provider is not yet implemented. "
            "Use 'gemini' provider for now."
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    def format_messages(self, messages: List[Message]) -> Any:
        """Convert to OpenAI chat format."""
        raise NotImplementedError

    def format_tools(self, tools: List[ToolDefinition]) -> Any:
        """Convert to OpenAI function format."""
        raise NotImplementedError

    def parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response."""
        raise NotImplementedError

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        raise NotImplementedError

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError
        yield  # Make it a generator
