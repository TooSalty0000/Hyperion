"""Conversation data models."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from shared.llm.types import Message


@dataclass
class Conversation:
    """
    A conversation with message history.

    Stores the full conversation context for an agent session,
    allowing conversation history to persist across interactions.
    """
    id: str
    user_id: Optional[int] = None
    channel_id: Optional[int] = None
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

    @property
    def message_count(self) -> int:
        """Get number of messages in conversation."""
        return len(self.messages)

    def add_message(self, message: Message):
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def get_recent_messages(self, count: int) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:] if count > 0 else []
