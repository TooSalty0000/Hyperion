"""Redis-backed conversation storage."""

import json
import uuid
import logging
from typing import Optional, List
from datetime import datetime

from shared.llm.types import Message, Role, ToolCall, ToolResult
from .models import Conversation

logger = logging.getLogger(__name__)

# Lazy import
_redis = None


def _import_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError(
                "redis is required for ConversationManager. "
                "Install it with: pip install redis"
            )
    return _redis


class ConversationManager:
    """
    Redis-backed conversation storage.

    Key patterns:
    - vega:conversation:{conv_id} - Conversation data (JSON)
    - vega:user:{user_id}:conversations - Set of user's conversation IDs
    - vega:channel:{channel_id}:conversation - Current conversation for channel
    """

    CONV_KEY = "vega:conversation:{conv_id}"
    USER_CONVS = "vega:user:{user_id}:conversations"
    CHANNEL_CONV = "vega:channel:{channel_id}:conversation"

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize conversation manager.

        Args:
            redis_url: Redis connection URL
        """
        self._redis_url = redis_url
        self._redis = None

    async def _get_client(self):
        """Get or create Redis client."""
        if self._redis is None:
            redis_async = _import_redis()
            self._redis = await redis_async.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )
        return self._redis

    def _serialize_message(self, msg: Message) -> dict:
        """Serialize a Message to dict."""
        d = {
            "role": msg.role.value,
            "content": msg.content
        }
        if msg.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ]
        if msg.tool_results:
            d["tool_results"] = [
                {
                    "call_id": tr.call_id,
                    "name": tr.name,
                    "result": tr.result,
                    "is_error": tr.is_error
                }
                for tr in msg.tool_results
            ]
        if msg.name:
            d["name"] = msg.name
        return d

    def _deserialize_message(self, d: dict) -> Message:
        """Deserialize a dict to Message."""
        tool_calls = None
        if "tool_calls" in d and d["tool_calls"]:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"]
                )
                for tc in d["tool_calls"]
            ]

        tool_results = None
        if "tool_results" in d and d["tool_results"]:
            tool_results = [
                ToolResult(
                    call_id=tr["call_id"],
                    name=tr["name"],
                    result=tr["result"],
                    is_error=tr.get("is_error", False)
                )
                for tr in d["tool_results"]
            ]

        return Message(
            role=Role(d["role"]),
            content=d.get("content"),
            tool_calls=tool_calls,
            tool_results=tool_results,
            name=d.get("name")
        )

    async def create(
        self,
        user_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Create a new conversation.

        Args:
            user_id: Discord user ID
            channel_id: Discord channel ID
            metadata: Optional metadata dict

        Returns:
            Conversation ID
        """
        client = await self._get_client()

        conv_id = str(uuid.uuid4())
        conv = Conversation(
            id=conv_id,
            user_id=user_id,
            channel_id=channel_id,
            metadata=metadata
        )

        key = self.CONV_KEY.format(conv_id=conv_id)
        await client.set(key, json.dumps({
            "id": conv.id,
            "user_id": conv.user_id,
            "channel_id": conv.channel_id,
            "messages": [],
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "metadata": conv.metadata
        }))

        # Index by user
        if user_id:
            await client.sadd(
                self.USER_CONVS.format(user_id=user_id),
                conv_id
            )

        # Set as current for channel
        if channel_id:
            await client.set(
                self.CHANNEL_CONV.format(channel_id=channel_id),
                conv_id
            )

        logger.debug(f"Created conversation {conv_id}")
        return conv_id

    async def get(self, conv_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        client = await self._get_client()
        key = self.CONV_KEY.format(conv_id=conv_id)

        data = await client.get(key)
        if not data:
            return None

        d = json.loads(data)
        messages = [self._deserialize_message(m) for m in d.get("messages", [])]

        return Conversation(
            id=d["id"],
            user_id=d.get("user_id"),
            channel_id=d.get("channel_id"),
            messages=messages,
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            metadata=d.get("metadata")
        )

    async def get_history(
        self,
        conv_id: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get conversation messages.

        Args:
            conv_id: Conversation ID
            limit: Optional limit on number of messages (from end)

        Returns:
            List of messages
        """
        conv = await self.get(conv_id)
        if not conv:
            return []

        messages = conv.messages
        if limit and limit > 0:
            messages = messages[-limit:]

        return messages

    async def save(
        self,
        conv_id: str,
        messages: List[Message],
        user_id: Optional[int] = None
    ):
        """
        Save/update conversation messages.

        Args:
            conv_id: Conversation ID
            messages: Full list of messages to save
            user_id: Optional user ID to associate
        """
        client = await self._get_client()
        key = self.CONV_KEY.format(conv_id=conv_id)

        data = await client.get(key)
        if data:
            conv_data = json.loads(data)
        else:
            conv_data = {
                "id": conv_id,
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat()
            }

        conv_data["messages"] = [
            self._serialize_message(m) for m in messages
        ]
        conv_data["updated_at"] = datetime.utcnow().isoformat()

        await client.set(key, json.dumps(conv_data))
        logger.debug(f"Saved {len(messages)} messages to conversation {conv_id}")

    async def append_message(self, conv_id: str, message: Message):
        """Append a single message to conversation."""
        messages = await self.get_history(conv_id)
        messages.append(message)
        await self.save(conv_id, messages)

    async def add_message(self, conv_id: str, role: str, content: str):
        """
        Add a message to conversation by role and content.

        Convenience method that creates a Message object internally.
        """
        message = Message(role=Role(role), content=content)
        await self.append_message(conv_id, message)

    async def get_channel_conversation(
        self,
        channel_id: int
    ) -> Optional[str]:
        """Get current conversation ID for a channel."""
        client = await self._get_client()
        return await client.get(
            self.CHANNEL_CONV.format(channel_id=channel_id)
        )

    async def set_channel_conversation(
        self,
        channel_id: int,
        conv_id: str
    ):
        """Set the current conversation for a channel."""
        client = await self._get_client()
        await client.set(
            self.CHANNEL_CONV.format(channel_id=channel_id),
            conv_id
        )

    async def delete(self, conv_id: str) -> bool:
        """Delete a conversation."""
        client = await self._get_client()
        conv = await self.get(conv_id)
        if not conv:
            return False

        # Remove from user index
        if conv.user_id:
            await client.srem(
                self.USER_CONVS.format(user_id=conv.user_id),
                conv_id
            )

        # Remove channel mapping if this is the current conversation
        if conv.channel_id:
            current = await self.get_channel_conversation(conv.channel_id)
            if current == conv_id:
                await client.delete(
                    self.CHANNEL_CONV.format(channel_id=conv.channel_id)
                )

        # Delete conversation
        key = self.CONV_KEY.format(conv_id=conv_id)
        await client.delete(key)

        logger.debug(f"Deleted conversation {conv_id}")
        return True

    async def list_user_conversations(
        self,
        user_id: int,
        limit: int = 10
    ) -> List[str]:
        """Get recent conversation IDs for a user."""
        client = await self._get_client()
        conv_ids = await client.smembers(
            self.USER_CONVS.format(user_id=user_id)
        )
        return list(conv_ids)[:limit]

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
