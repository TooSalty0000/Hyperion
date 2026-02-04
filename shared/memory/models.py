"""Memory data models for agent persistent memory."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class MemoryType(Enum):
    """Memory storage type."""
    SHORT = "short"
    LONG = "long"


# Configuration constants
SHORT_TERM_MAX_COUNT = 20
SHORT_TERM_MAX_TOKENS = 2000
LONG_TERM_SUMMARY_MAX_CHARS = 200
IMPORTANCE_PROMOTION_THRESHOLD = 0.8
IMPORTANCE_DECAY_RATE = 0.05
AUTO_SELECT_LIMIT = 5


@dataclass
class MemoryEntry:
    """
    A single memory entry.

    Stores information the agent wants to remember across conversations.
    Can be short-term (always in context) or long-term (retrieved on demand).
    """
    id: str                                    # UUID
    agent_id: str                              # "vega" or "altair"
    content: str                               # The memory content
    summary: Optional[str] = None              # Summarized version (for long-term)

    # Classification
    memory_type: MemoryType = MemoryType.SHORT # "short" or "long"
    category: Optional[str] = None             # For long-term organization
    importance: float = 0.5                    # 0.0-1.0, agent-assigned

    # Metadata
    source_conversation_id: Optional[str] = None
    source_message_excerpt: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    # Context for retrieval
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for Redis storage."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "summary": self.summary,
            "memory_type": self.memory_type.value,
            "category": self.category,
            "importance": self.importance,
            "source_conversation_id": self.source_conversation_id,
            "source_message_excerpt": self.source_message_excerpt,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            content=data["content"],
            summary=data.get("summary"),
            memory_type=MemoryType(data.get("memory_type", "short")),
            category=data.get("category"),
            importance=data.get("importance", 0.5),
            source_conversation_id=data.get("source_conversation_id"),
            source_message_excerpt=data.get("source_message_excerpt"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            accessed_at=datetime.fromisoformat(data["accessed_at"]) if data.get("accessed_at") else datetime.utcnow(),
            access_count=data.get("access_count", 0),
            keywords=data.get("keywords", []),
        )

    def format_for_prompt(self, include_category: bool = False) -> str:
        """Format memory for inclusion in agent prompt."""
        importance_str = f"[{self.importance:.1f}]"
        if include_category and self.category:
            return f"- {importance_str} [{self.category}] {self.summary or self.content}"
        return f"- {importance_str} {self.content}"


@dataclass
class MemoryCategory:
    """
    Category for organizing long-term memories.

    Categories are auto-created when memories are promoted to long-term.
    """
    id: str                                    # Category identifier (slug)
    agent_id: str
    name: str                                  # Display name
    description: str                           # What this category stores
    memory_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for Redis storage."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "memory_count": self.memory_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryCategory':
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            name=data["name"],
            description=data["description"],
            memory_count=data.get("memory_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )


@dataclass
class MemoryContext:
    """
    Context provided to agent prompts about available memories.

    Built by MemoryManager.build_memory_context() and formatted for prompt injection.
    """
    short_term_memories: List[MemoryEntry] = field(default_factory=list)
    category_summaries: List[Dict[str, Any]] = field(default_factory=list)
    auto_selected_memories: List[MemoryEntry] = field(default_factory=list)
    recently_accessed_ids: List[str] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Format complete memory context as prompt text."""
        sections = []

        # Short-term memories (always included)
        if self.short_term_memories:
            lines = ["### ACTIVE CONTEXT (Always Remember):"]
            # Sort by importance descending
            sorted_memories = sorted(
                self.short_term_memories,
                key=lambda m: m.importance,
                reverse=True
            )
            for memory in sorted_memories:
                lines.append(memory.format_for_prompt())
            sections.append("\n".join(lines))

        # Category summaries (for awareness)
        if self.category_summaries:
            lines = ["### LONG-TERM CATEGORIES:"]
            lines.append("Use search_memories to retrieve from these:")
            for cat in self.category_summaries:
                lines.append(f"- **{cat['name']}** ({cat['count']} memories): {cat['description']}")
            sections.append("\n".join(lines))

        # Auto-selected relevant memories
        if self.auto_selected_memories:
            lines = ["### AUTO-SELECTED (Relevant to current context):"]
            for memory in self.auto_selected_memories:
                lines.append(memory.format_for_prompt(include_category=True))
            sections.append("\n".join(lines))

        if not sections:
            return "No memories stored yet. Use store_memory to remember important information."

        return "\n\n".join(sections)


# Default categories that agents can use
DEFAULT_CATEGORIES = {
    "user_preferences": "User communication style, preferences, and work habits",
    "project_context": "Information about active projects and their structure",
    "technical_notes": "Technical details, solutions, and patterns discovered",
    "conversation_highlights": "Important moments from past conversations",
    "workflow_patterns": "Agent's learned workflows and effective approaches",
    "passive_context": "Summarized observations from passively monitored chat messages",
}
