"""Ollama LLM provider implementation using the official ollama Python package."""

import asyncio
import json
import logging
import os
from typing import List, Optional, AsyncIterator, Any, Dict

from .interface import LLMProvider
from .types import (
    Message, Role, LLMResponse, StreamChunk,
    ToolDefinition, ToolCall
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 15.0  # seconds

DEFAULT_BASE_URL = "http://localhost:11434"

# Lazy import to avoid requiring ollama if not used
_ollama = None


def _import_ollama():
    """Lazy import ollama package."""
    global _ollama
    if _ollama is None:
        try:
            import ollama as ollama_pkg
            _ollama = ollama_pkg
        except ImportError:
            raise ImportError(
                "ollama is required for Ollama provider. "
                "Install it with: pip install ollama"
            )
    return _ollama


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (transient connection issue)."""
    error_str = str(error).lower()
    retryable_patterns = [
        "connection refused",
        "connection reset",
        "connection error",
        "timeout",
        "temporarily unavailable",
        "server disconnected",
        "broken pipe",
        "eof",
        "incomplete read",
    ]
    return any(pattern in error_str for pattern in retryable_patterns)


def _is_tool_unsupported_error(error: Exception) -> bool:
    """Check if an error indicates the model doesn't support tool calling."""
    error_str = str(error).lower()
    unsupported_patterns = [
        "does not support tools",
        "tool use is not supported",
        "tools are not supported",
        "unknown parameter: tools",
        "does not support function",
    ]
    return any(pattern in error_str for pattern in unsupported_patterns)


# Template for prompt-based tool calling fallback
TOOL_PROMPT_TEMPLATE = """You have access to tools. Respond with a tool_calls array containing exactly ONE tool call.

Available tools:
{tool_descriptions}

RULES:
- Call exactly ONE tool at a time.
- The "arguments" object MUST match the tool's parameter schema exactly.
- Every required field must be present."""


def _build_tool_prompt(tools: List[ToolDefinition]) -> str:
    """Build a text description of tools for prompt-based fallback."""
    descriptions = []
    for tool in tools:
        schema = tool.to_json_schema()
        # Compact schema representation for the prompt
        desc = f"- {tool.name}: {tool.description}\n  Args: {json.dumps(schema)}"
        descriptions.append(desc)

    return TOOL_PROMPT_TEMPLATE.format(tool_descriptions="\n".join(descriptions))


def _build_response_schema(tools: List[ToolDefinition]) -> Dict[str, Any]:
    """
    Build a JSON schema for Ollama's structured output (format parameter).

    Constrains the model to output a valid tool call with known tool names.
    """
    tool_names = [tool.name for tool in tools]

    return {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": tool_names
                        },
                        "arguments": {"type": "object"}
                    },
                    "required": ["name", "arguments"]
                }
            }
        },
        "required": ["tool_calls"]
    }


def _parse_prompt_tool_response(text: str) -> tuple[Optional[List[ToolCall]], Optional[str]]:
    """
    Parse a response from prompt-based tool calling.

    The model is expected to return either:
    - {"tool_calls": [{"name": "...", "arguments": {...}}]}
    - {"message": "response text"}

    Returns:
        Tuple of (tool_calls, message_content).
        - If tool calls found: (calls, None)
        - If message found: (None, message_text)
        - If parse fails: (None, None)
    """
    if not text:
        return None, None

    # Try to find JSON in the response (may be wrapped in markdown code blocks)
    content = text.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            # Check for tool calls
            if "tool_calls" in parsed:
                calls = []
                for i, tc in enumerate(parsed["tool_calls"]):
                    if isinstance(tc, dict) and "name" in tc:
                        calls.append(ToolCall(
                            id=f"call_{i}",
                            name=tc["name"],
                            arguments=tc.get("arguments", {})
                        ))
                if calls:
                    return calls, None

            # Check for message response (model chose not to use tools)
            if "message" in parsed:
                return None, str(parsed["message"])

    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    return None, None


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for local/remote model inference.

    Supports native tool calling for models that have it, and gracefully
    degrades to prompt-based tool calling for models that don't.

    Configuration:
        - base_url: Ollama server URL (kwarg, env OLLAMA_BASE_URL, or default localhost:11434)
        - api_key: Ignored for local Ollama (kept for interface compatibility)
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemma3:4b",
        **kwargs
    ):
        """
        Initialize Ollama provider.

        Args:
            api_key: Unused for local Ollama (interface compatibility)
            model: Ollama model name (default: gemma3:4b)
            **kwargs: Additional configuration:
                - base_url: Ollama server URL (default: http://localhost:11434)
        """
        super().__init__(api_key, model, **kwargs)

        # Resolve base URL: kwargs > env > default
        self._base_url = (
            kwargs.get("base_url")
            or os.getenv("OLLAMA_BASE_URL")
            or DEFAULT_BASE_URL
        )

        # Lazy-initialized client
        self._client = None

        # Track whether this model supports native tool calling
        # None = unknown (not yet tested), True/False = known
        self._model_supports_tools: Optional[bool] = None

    def _get_client(self):
        """Get or create the async Ollama client."""
        if self._client is None:
            ollama = _import_ollama()
            self._client = ollama.AsyncClient(host=self._base_url)
        return self._client

    @property
    def provider_name(self) -> str:
        return "ollama"

    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert unified messages to Ollama's chat format.

        Ollama uses the OpenAI-style message format:
        [{"role": "system|user|assistant|tool", "content": "...", ...}]
        """
        formatted = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                formatted.append({
                    "role": "system",
                    "content": msg.content or ""
                })

            elif msg.role == Role.USER:
                formatted.append({
                    "role": "user",
                    "content": msg.content or ""
                })

            elif msg.role == Role.ASSISTANT:
                entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or ""
                }
                # Include tool calls if present (for multi-turn context)
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                formatted.append(entry)

            elif msg.role == Role.TOOL and msg.tool_results:
                # Each tool result becomes a separate tool message
                for tr in msg.tool_results:
                    formatted.append({
                        "role": "tool",
                        "content": tr.result
                    })

        return formatted

    def format_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """
        Convert unified tools to Ollama's format (OpenAI-compatible).

        Returns list of tool definitions in the format:
        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
        """
        formatted = []
        for tool in tools:
            schema = tool.to_json_schema()
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema
                }
            })
        return formatted

    def parse_response(self, response: Any) -> LLMResponse:
        """
        Parse Ollama response to unified format.

        Ollama chat response has:
        - message.content: text content
        - message.tool_calls: list of tool calls (if model supports it)
        - prompt_eval_count / eval_count: token usage
        """
        content = None
        tool_calls = None
        raw_content = None

        message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", None)

        if message:
            # Handle both dict and object access patterns
            if isinstance(message, dict):
                msg_content = message.get("content", "")
                msg_tool_calls = message.get("tool_calls")
            else:
                msg_content = getattr(message, "content", "")
                msg_tool_calls = getattr(message, "tool_calls", None)

            if msg_content:
                content = msg_content.strip()

            if msg_tool_calls:
                tool_calls = []
                for i, tc in enumerate(msg_tool_calls):
                    # Handle both dict and object formats
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        args = func.get("arguments", {})
                    else:
                        func = getattr(tc, "function", None)
                        if func:
                            name = getattr(func, "name", "") if not isinstance(func, dict) else func.get("name", "")
                            args = getattr(func, "arguments", {}) if not isinstance(func, dict) else func.get("arguments", {})
                        else:
                            name = getattr(tc, "name", "")
                            args = getattr(tc, "arguments", {})

                    # Arguments may be a JSON string that needs parsing
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}

                    if name:
                        tool_calls.append(ToolCall(
                            id=f"call_{i}",
                            name=name,
                            arguments=args if isinstance(args, dict) else {}
                        ))

                # Preserve raw for multi-turn
                if tool_calls:
                    raw_content = message

        # Determine finish reason
        if tool_calls:
            finish_reason = "tool_calls"
        else:
            # Check if response was truncated
            done = response.get("done", True) if isinstance(response, dict) else getattr(response, "done", True)
            done_reason = response.get("done_reason", "") if isinstance(response, dict) else getattr(response, "done_reason", "")
            if not done or done_reason == "length":
                finish_reason = "length"
            else:
                finish_reason = "stop"

        # Extract token usage
        usage = None
        if isinstance(response, dict):
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
        else:
            prompt_tokens = getattr(response, "prompt_eval_count", 0)
            completion_tokens = getattr(response, "eval_count", 0)

        if prompt_tokens or completion_tokens:
            usage = {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0)
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
            raw_content=raw_content
        )

    async def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion from Ollama with retry logic and tool calling fallback.

        If the model doesn't support native tool calling, falls back to
        prompt-based tool descriptions injected into the system message.
        """
        client = self._get_client()
        formatted_messages = self.format_messages(messages)

        # Determine whether to use native tools or prompt-based fallback
        use_native_tools = tools and self._model_supports_tools is not False
        use_prompt_tools = tools and self._model_supports_tools is False

        # For prompt-based fallback, inject tool descriptions into system message
        if use_prompt_tools:
            tool_prompt = _build_tool_prompt(tools)
            # Prepend or append to existing system message
            if formatted_messages and formatted_messages[0].get("role") == "system":
                formatted_messages[0]["content"] += "\n\n" + tool_prompt
            else:
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": tool_prompt
                })

        # Build request kwargs
        chat_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "options": {"temperature": temperature},
        }

        if max_tokens:
            chat_kwargs["options"]["num_predict"] = max_tokens

        if use_native_tools:
            chat_kwargs["tools"] = self.format_tools(tools)

        # Structured output format:
        # - Explicit format from kwargs (e.g., "json" or a schema dict)
        # - For prompt-based tool fallback, use a response schema that constrains
        #   output to valid tool calls with known tool names
        if kwargs.get("format"):
            chat_kwargs["format"] = kwargs["format"]
        elif use_prompt_tools:
            chat_kwargs["format"] = _build_response_schema(tools)

        last_error = None
        backoff = INITIAL_BACKOFF

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.chat(**chat_kwargs)

                # If we were unsure about tool support and it worked, mark as supported
                if use_native_tools and self._model_supports_tools is None:
                    self._model_supports_tools = True

                result = self.parse_response(response)

                # For prompt-based fallback, parse the JSON response
                if use_prompt_tools and result.content and not result.tool_calls:
                    parsed_calls, parsed_message = _parse_prompt_tool_response(result.content)
                    if parsed_calls:
                        result.tool_calls = parsed_calls
                        result.finish_reason = "tool_calls"
                        result.content = None
                    elif parsed_message is not None:
                        # Model responded with {"message": "..."} — extract it
                        result.content = parsed_message

                return result

            except Exception as e:
                last_error = e

                # Check if this is a tool-unsupported error
                if use_native_tools and _is_tool_unsupported_error(e):
                    logger.warning(
                        f"Model '{self.model}' does not support native tool calling. "
                        f"Falling back to prompt-based tools."
                    )
                    self._model_supports_tools = False

                    # Retry immediately with prompt-based fallback
                    return await self.complete(
                        messages, tools, temperature, max_tokens, **kwargs
                    )

                # Check if retryable
                if attempt < MAX_RETRIES and _is_retryable_error(e):
                    logger.warning(
                        f"Ollama API error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue

                # Non-retryable error or max retries exceeded
                logger.error(f"Ollama API error (final): {e}", exc_info=True)
                break

        error_msg = str(last_error) if last_error else "Unknown error"
        raise RuntimeError(f"Ollama API failed after {MAX_RETRIES + 1} attempts: {error_msg}")

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion from Ollama.

        Note: Streaming with tool calling is limited. If tools are provided
        and the model supports them, tool calls will arrive in the final chunk.
        For prompt-based fallback, the full response is accumulated and parsed.
        """
        client = self._get_client()
        formatted_messages = self.format_messages(messages)

        # For streaming with prompt-based tools, inject descriptions
        use_prompt_tools = tools and self._model_supports_tools is False
        if use_prompt_tools:
            tool_prompt = _build_tool_prompt(tools)
            if formatted_messages and formatted_messages[0].get("role") == "system":
                formatted_messages[0]["content"] += "\n\n" + tool_prompt
            else:
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": tool_prompt
                })

        chat_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        if max_tokens:
            chat_kwargs["options"]["num_predict"] = max_tokens

        # Only pass native tools if model supports them
        use_native_tools = tools and self._model_supports_tools is not False
        if use_native_tools:
            chat_kwargs["tools"] = self.format_tools(tools)

        # Structured output for prompt-based fallback
        if kwargs.get("format"):
            chat_kwargs["format"] = kwargs["format"]
        elif use_prompt_tools:
            chat_kwargs["format"] = _build_response_schema(tools)

        try:
            accumulated_content = ""

            async for chunk in await client.chat(**chat_kwargs):
                # Handle both dict and object response formats
                if isinstance(chunk, dict):
                    message = chunk.get("message", {})
                    done = chunk.get("done", False)
                else:
                    message = getattr(chunk, "message", None)
                    done = getattr(chunk, "done", False)

                if message:
                    if isinstance(message, dict):
                        chunk_content = message.get("content", "")
                        chunk_tool_calls = message.get("tool_calls")
                    else:
                        chunk_content = getattr(message, "content", "")
                        chunk_tool_calls = getattr(message, "tool_calls", None)

                    if chunk_content:
                        accumulated_content += chunk_content
                        yield StreamChunk(content=chunk_content)

                    # Tool calls typically arrive in the final chunk
                    if chunk_tool_calls and done:
                        for i, tc in enumerate(chunk_tool_calls):
                            if isinstance(tc, dict):
                                func = tc.get("function", {})
                            else:
                                func = getattr(tc, "function", {})
                                if not isinstance(func, dict):
                                    func = {"name": getattr(func, "name", ""), "arguments": getattr(func, "arguments", {})}

                            yield StreamChunk(
                                tool_call_delta={
                                    "id": f"call_{i}",
                                    "name": func.get("name", ""),
                                    "arguments": func.get("arguments", {})
                                },
                                finish_reason="tool_calls"
                            )
                            return

                if done:
                    break

            # For prompt-based fallback, parse accumulated JSON response
            if use_prompt_tools and accumulated_content:
                parsed_calls, parsed_message = _parse_prompt_tool_response(accumulated_content)
                if parsed_calls:
                    for i, tc in enumerate(parsed_calls):
                        yield StreamChunk(
                            tool_call_delta={
                                "id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments
                            },
                            finish_reason="tool_calls"
                        )
                    return

            yield StreamChunk(finish_reason="stop")

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}", exc_info=True)
            yield StreamChunk(content=f"Error: {str(e)}", finish_reason="error")
