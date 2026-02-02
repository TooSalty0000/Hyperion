"""Google Gemini API provider implementation using google-genai SDK."""

import asyncio
import json
import logging
import time
from typing import List, Optional, AsyncIterator, Any, Tuple

from .interface import LLMProvider
from .types import (
    Message, Role, LLMResponse, StreamChunk,
    ToolDefinition, ToolCall
)

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 30.0  # seconds
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Lazy import to avoid requiring google-genai if not used
_genai = None
_types = None


def _import_genai():
    """Lazy import google.genai."""
    global _genai, _types
    if _genai is None:
        try:
            from google import genai
            from google.genai import types
            _genai = genai
            _types = types
        except ImportError:
            raise ImportError(
                "google-genai is required for Gemini provider. "
                "Install it with: pip install google-genai"
            )
    return _genai, _types


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (transient)."""
    error_str = str(error).lower()

    # Check for HTTP status codes in error message
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in error_str:
            return True

    # Check for common transient error patterns
    retryable_patterns = [
        "rate limit",
        "quota exceeded",
        "too many requests",
        "server error",
        "service unavailable",
        "internal error",
        "temporarily unavailable",
        "overloaded",
        "capacity",
        "timeout",
        "connection reset",
        "connection refused",
    ]

    return any(pattern in error_str for pattern in retryable_patterns)


# Patterns that indicate prompt leakage
PROMPT_LEAK_PATTERNS = [
    # System reminder tags
    "<system-reminder>",
    "</system-reminder>",
    "system-reminder",
    # Generic prompt markers
    "CRITICAL:",
    "IMPORTANT:",
    "ROLE:",
    "WORKFLOW:",
    "BOUNDARIES:",
    "BEST PRACTICES:",
    "COMMUNICATION STYLE:",
    "BROWSER TOOLS:",
    "WHEN MENTIONED BY",
    "AVAILABLE TOOLS:",
    "CURRENT STATE:",
    "YOUR MEMORIES:",
    # File/path leaks
    "Note: /Users/",
    "The user opened the file",
    # Tool references
    "This is a reminder that",
    "TodoWrite tool",
    # Response formatting instructions (Canopus-specific)
    "Stay in character as",
    "Keep response under",
    "Do NOT use",
    "NEVER repeat the same",
    "NEVER prefix your responses",
    "NEVER prefix messages with",
    "Provide a concise final response",
    # Identity/persona instructions
    "=== IDENTITY:",
    "PERSONALITY:",
    "UNDERSTANDING CHAT HISTORY:",
    "RESPONSE FORMAT:",
    "AVOID DUPLICATE/REPETITIVE",
    # Agent boundaries
    "You are ONLY",
    "You are NOT Vega",
    "You are NOT Altair",
    "You are NOT Polaris",
    "ALWAYS be Canopus",
    "Discord already shows who is speaking",
]

# Structured output schema to prevent prompt leakage
# LLM must return JSON with a "message" field
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "message": {
            "type": "STRING",
            "description": "The response message to send to the user"
        }
    },
    "required": ["message"]
}

# Schema for chime-in decision (utility LLM)
CHIME_IN_DECISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "decision": {
            "type": "STRING",
            "enum": ["RESPOND", "STAY_QUIET", "NOT_RELEVANT"],
            "description": "Whether to chime in on this message"
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence level from 0.0 to 1.0"
        },
        "reasoning": {
            "type": "STRING",
            "description": "Brief explanation for the decision"
        }
    },
    "required": ["decision", "confidence", "reasoning"]
}


def _clean_prompt_leakage(content: str) -> str:
    """Remove prompt leakage from LLM response."""
    if not content:
        return content

    original_len = len(content)
    lines = content.split('\n')
    cleaned_lines = []
    skip_until_close_tag = False
    removed_count = 0

    for line in lines:
        # Skip content inside system-reminder tags
        if "<system-reminder>" in line:
            skip_until_close_tag = True
            removed_count += 1
            continue
        if "</system-reminder>" in line:
            skip_until_close_tag = False
            removed_count += 1
            continue
        if skip_until_close_tag:
            removed_count += 1
            continue

        # Check for obvious prompt leak patterns
        line_stripped = line.strip()
        is_leak = False

        for pattern in PROMPT_LEAK_PATTERNS:
            if pattern in line:
                is_leak = True
                break

        # Also check for lines that look like file paths from system context
        if line_stripped.startswith("/Users/") or line_stripped.startswith("Note: /"):
            is_leak = True

        if is_leak:
            removed_count += 1
        else:
            cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines)

    # Clean up excessive whitespace from removals
    while '\n\n\n' in result:
        result = result.replace('\n\n\n', '\n\n')

    result = result.strip()

    # Log if we cleaned significant content
    if removed_count > 0:
        logger.warning(
            f"Cleaned prompt leakage: removed {removed_count} lines, "
            f"reduced from {original_len} to {len(result)} chars"
        )

    return result


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        **kwargs
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google AI API key
            model: Model name (default: gemini-2.0-flash-exp)
            **kwargs: Additional configuration
        """
        super().__init__(api_key, model, **kwargs)
        genai, _ = _import_genai()
        self._client = genai.Client(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def format_messages(self, messages: List[Message]) -> Tuple[List[Any], Optional[str]]:
        """
        Convert to Gemini's content format.

        Returns:
            Tuple of (contents list, system instruction or None)
        """
        _, types = _import_genai()
        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
                continue

            if msg.role == Role.USER:
                # Simple string content for user messages
                if msg.content:
                    contents.append(msg.content)

            elif msg.role == Role.ASSISTANT:
                # For assistant messages, use raw_content if available (preserves thought_signature)
                if msg.raw_content is not None:
                    contents.append(msg.raw_content)
                elif msg.content:
                    # Text content from assistant - wrap as model content
                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=msg.content)]
                    ))

            elif msg.role == Role.TOOL and msg.tool_results:
                # Function responses
                for tr in msg.tool_results:
                    contents.append(types.Part.from_function_response(
                        name=tr.name,
                        response={"result": tr.result}
                    ))

        return contents, system_instruction

    def format_tools(self, tools: List[ToolDefinition]) -> List[Any]:
        """Convert to Gemini's function declarations."""
        _, types = _import_genai()

        declarations = []
        for tool in tools:
            schema = tool.to_json_schema()
            # Use parameters_json_schema as per the API
            declarations.append(types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=schema if schema.get("properties") else None
            ))

        return [types.Tool(function_declarations=declarations)] if declarations else None

    def parse_response(self, response: Any) -> LLMResponse:
        """Parse Gemini response to unified format."""
        content = None
        tool_calls = []
        raw_content = None

        # Extract text content directly from parts to avoid warning
        # when response contains both text and function_call parts
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_parts = []
                for part in candidate.content.parts:
                    # Check for text content
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    # Check for function calls
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_calls.append(ToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {}
                        ))

                if text_parts:
                    content = "".join(text_parts)

                    # Parse structured output JSON to extract message
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "message" in parsed:
                            content = parsed["message"]
                    except (json.JSONDecodeError, TypeError, ValueError):
                        # Not valid JSON - fall back to pattern-based cleaning
                        logger.debug("Response not valid JSON, using pattern cleaning")
                        content = _clean_prompt_leakage(content)

                # Preserve raw content for multi-turn (includes thought_signature)
                if tool_calls:
                    raw_content = candidate.content

        finish_reason = "tool_calls" if tool_calls else "stop"

        # Extract usage if available
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, 'prompt_token_count', 0),
                "completion_tokens": getattr(um, 'candidates_token_count', 0),
                "total_tokens": getattr(um, 'total_token_count', 0)
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
        """Generate a completion from Gemini with retry logic."""
        start_time = time.time()
        _, types = _import_genai()

        format_start = time.time()
        contents, system = self.format_messages(messages)
        tool_config = self.format_tools(tools) if tools else None
        format_time = (time.time() - format_start) * 1000

        # Allow custom response schema via kwargs (e.g., for chime-in decisions)
        response_schema = kwargs.get("response_schema", RESPONSE_SCHEMA)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 8192,
            system_instruction=system,
            tools=tool_config,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        last_error = None
        backoff = INITIAL_BACKOFF
        tool_count = len(tools) if tools else 0
        msg_count = len(messages)

        logger.debug(
            f"[LLM] Starting request: messages={msg_count}, tools={tool_count}, "
            f"format_time={format_time:.1f}ms"
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                api_start = time.time()
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )
                api_time = (time.time() - api_start) * 1000
                total_time = (time.time() - start_time) * 1000

                parsed = self.parse_response(response)

                # Log timing details
                tool_calls = len(parsed.tool_calls) if parsed.tool_calls else 0
                logger.debug(
                    f"[LLM] ✓ Complete: api={api_time:.0f}ms, total={total_time:.0f}ms, "
                    f"tool_calls={tool_calls}, content_len={len(parsed.content) if parsed.content else 0}"
                )

                return parsed

            except Exception as e:
                last_error = e

                # Check if error is retryable
                if attempt < MAX_RETRIES and _is_retryable_error(e):
                    logger.warning(
                        f"Gemini API error (attempt {attempt + 1}/{MAX_RETRIES + 1}): {e}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)  # Exponential backoff
                    continue

                # Non-retryable error or max retries exceeded
                logger.error(f"Gemini API error (final): {e}", exc_info=True)
                break

        # All retries exhausted or non-retryable error
        error_msg = str(last_error) if last_error else "Unknown error"
        raise RuntimeError(f"Gemini API failed after {MAX_RETRIES + 1} attempts: {error_msg}")

    async def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from Gemini."""
        _, types = _import_genai()

        contents, system = self.format_messages(messages)
        tool_config = self.format_tools(tools) if tools else None

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 8192,
            system_instruction=system,
            tools=tool_config
        )

        try:
            async for chunk in self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config
            ):
                try:
                    if chunk.text:
                        yield StreamChunk(content=chunk.text)
                except Exception:
                    pass

            yield StreamChunk(finish_reason="stop")

        except Exception as e:
            logger.error(f"Gemini streaming error: {e}", exc_info=True)
            yield StreamChunk(content=f"Error: {str(e)}", finish_reason="error")
