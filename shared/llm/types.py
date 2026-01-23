"""Shared types for LLM providers."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class Role(Enum):
    """Message role in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolParameter:
    """Single parameter definition for a tool."""
    name: str
    type: str  # "string", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None
    # For array types: schema of each item (JSON Schema dict)
    items: Optional[Dict[str, Any]] = None
    # For object types: nested properties (JSON Schema dict)
    properties: Optional[Dict[str, Any]] = None


@dataclass
class ToolDefinition:
    """Provider-agnostic tool definition."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format (used by most providers)."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            if param.items:
                prop["items"] = param.items
            if param.properties:
                prop["properties"] = param.properties
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required
        }


@dataclass
class ToolCall:
    """A tool call made by the LLM."""
    id: str  # Provider-specific call ID
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool."""
    call_id: str
    name: str
    result: str  # Always string for LLM consumption
    is_error: bool = False


@dataclass
class Message:
    """A message in the conversation."""
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None
    name: Optional[str] = None  # For tool results
    raw_content: Optional[Any] = None  # Provider-specific raw content (for passthrough)


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str = "stop"  # "stop", "tool_calls", "length", "error"
    usage: Optional[Dict[str, int]] = None  # tokens used
    raw_content: Optional[Any] = None  # Provider-specific raw content (for multi-turn)


@dataclass
class StreamChunk:
    """A chunk of streamed response."""
    content: Optional[str] = None
    tool_call_delta: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
