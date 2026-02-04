"""Redis-backed memory storage for agents."""

import asyncio
import json
import uuid
import re
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from .models import (
    MemoryEntry,
    MemoryCategory,
    MemoryContext,
    MemoryType,
    SHORT_TERM_MAX_COUNT,
    AUTO_SELECT_LIMIT,
    DEFAULT_CATEGORIES,
)

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
                "redis is required for MemoryManager. "
                "Install it with: pip install redis"
            )
    return _redis


class MemoryManager:
    """
    Redis-backed memory storage for agents.

    Provides persistent memory that agents can use across conversations.
    Supports short-term (always in context) and long-term (retrieved on demand) memories.

    Key patterns:
    - hyperion:memory:{agent_id}:entry:{memory_id} - Memory data (JSON)
    - hyperion:memory:{agent_id}:short - Set of short-term memory IDs
    - hyperion:memory:{agent_id}:long - Set of long-term memory IDs
    - hyperion:memory:{agent_id}:category:{category_id} - Category data (JSON)
    - hyperion:memory:{agent_id}:categories - Set of category IDs
    - hyperion:memory:{agent_id}:category:{category_id}:memories - Set of memory IDs in category
    - hyperion:memory:{agent_id}:by_importance - Sorted set: memory_id -> importance
    - hyperion:memory:{agent_id}:keywords:{keyword} - Set of memory IDs with keyword
    """

    # Key patterns
    ENTRY_KEY = "hyperion:memory:{agent_id}:entry:{memory_id}"
    SHORT_SET = "hyperion:memory:{agent_id}:short"
    LONG_SET = "hyperion:memory:{agent_id}:long"
    CATEGORY_KEY = "hyperion:memory:{agent_id}:category:{category_id}"
    CATEGORIES_SET = "hyperion:memory:{agent_id}:categories"
    CATEGORY_MEMORIES = "hyperion:memory:{agent_id}:category:{category_id}:memories"
    BY_IMPORTANCE = "hyperion:memory:{agent_id}:by_importance"
    KEYWORD_SET = "hyperion:memory:{agent_id}:keywords:{keyword}"

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize memory manager.

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

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def store_memory(
        self,
        agent_id: str,
        content: str,
        importance: float = 0.5,
        category: Optional[str] = None,
        source_conversation_id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> MemoryEntry:
        """
        Store a new memory. Initially placed in short-term.

        Args:
            agent_id: Agent identifier ("vega" or "altair")
            content: The information to remember
            importance: Importance score 0.0-1.0 (higher = more important)
            category: Optional category for organization
            source_conversation_id: ID of conversation that generated this memory
            keywords: Keywords for search (auto-extracted if not provided)

        Returns:
            The created MemoryEntry
        """
        redis = await self._get_client()

        # Generate ID and extract keywords if not provided
        memory_id = str(uuid.uuid4())
        if keywords is None:
            keywords = self._extract_keywords(content)

        # Create memory entry
        memory = MemoryEntry(
            id=memory_id,
            agent_id=agent_id,
            content=content,
            memory_type=MemoryType.SHORT,
            category=category,
            importance=max(0.0, min(1.0, importance)),  # Clamp to 0-1
            source_conversation_id=source_conversation_id,
            keywords=keywords,
        )

        # Store in Redis
        entry_key = self.ENTRY_KEY.format(agent_id=agent_id, memory_id=memory_id)
        await redis.set(entry_key, json.dumps(memory.to_dict()))

        # Add to short-term set
        short_key = self.SHORT_SET.format(agent_id=agent_id)
        await redis.sadd(short_key, memory_id)

        # Add to importance sorted set
        importance_key = self.BY_IMPORTANCE.format(agent_id=agent_id)
        await redis.zadd(importance_key, {memory_id: importance})

        # Index keywords
        for keyword in keywords:
            kw_key = self.KEYWORD_SET.format(agent_id=agent_id, keyword=keyword.lower())
            await redis.sadd(kw_key, memory_id)

        # Enforce short-term limits
        await self._enforce_short_term_limits(agent_id)

        logger.info(f"Stored memory {memory_id} for {agent_id}: {content[:50]}...")
        return memory

    async def get_memory(self, agent_id: str, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory by ID.

        Also updates access metadata.
        """
        redis = await self._get_client()

        entry_key = self.ENTRY_KEY.format(agent_id=agent_id, memory_id=memory_id)
        data = await redis.get(entry_key)

        if not data:
            return None

        memory = MemoryEntry.from_dict(json.loads(data))

        # Update access metadata
        memory.accessed_at = datetime.utcnow()
        memory.access_count += 1
        await redis.set(entry_key, json.dumps(memory.to_dict()))

        return memory

    async def get_memory_readonly(
        self, agent_id: str, memory_id: str
    ) -> Optional[MemoryEntry]:
        """
        Retrieve a memory by ID without updating access metadata.

        Use this for read-only context building to avoid unnecessary writes.
        """
        redis = await self._get_client()

        entry_key = self.ENTRY_KEY.format(agent_id=agent_id, memory_id=memory_id)
        data = await redis.get(entry_key)

        if not data:
            return None

        return MemoryEntry.from_dict(json.loads(data))

    async def get_memories_batch(
        self,
        agent_id: str,
        memory_ids: List[str],
    ) -> List[MemoryEntry]:
        """
        Fetch multiple memories in a single mget call.

        This is a read-only operation that doesn't update access metadata.
        Use for context building to avoid N+1 Redis calls.

        Args:
            agent_id: Agent identifier
            memory_ids: List of memory IDs to fetch

        Returns:
            List of MemoryEntry objects (excludes any not found)
        """
        if not memory_ids:
            return []

        redis = await self._get_client()
        keys = [
            self.ENTRY_KEY.format(agent_id=agent_id, memory_id=mid)
            for mid in memory_ids
        ]

        results = await redis.mget(keys)

        memories = []
        for data in results:
            if data:
                memories.append(MemoryEntry.from_dict(json.loads(data)))

        return memories

    async def update_memory(
        self,
        agent_id: str,
        memory_id: str,
        importance: Optional[float] = None,
        category: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> bool:
        """
        Update memory metadata.

        Args:
            agent_id: Agent identifier
            memory_id: Memory to update
            importance: New importance score (if provided)
            category: New category (if provided)
            keywords: New keywords (if provided)

        Returns:
            True if updated, False if memory not found
        """
        redis = await self._get_client()

        entry_key = self.ENTRY_KEY.format(agent_id=agent_id, memory_id=memory_id)
        data = await redis.get(entry_key)

        if not data:
            return False

        memory = MemoryEntry.from_dict(json.loads(data))

        # Update fields
        if importance is not None:
            old_importance = memory.importance
            memory.importance = max(0.0, min(1.0, importance))

            # Update sorted set
            importance_key = self.BY_IMPORTANCE.format(agent_id=agent_id)
            await redis.zadd(importance_key, {memory_id: memory.importance})

        if category is not None:
            memory.category = category

        if keywords is not None:
            # Remove old keyword indexes
            for kw in memory.keywords:
                kw_key = self.KEYWORD_SET.format(agent_id=agent_id, keyword=kw.lower())
                await redis.srem(kw_key, memory_id)

            # Add new keyword indexes
            for kw in keywords:
                kw_key = self.KEYWORD_SET.format(agent_id=agent_id, keyword=kw.lower())
                await redis.sadd(kw_key, memory_id)

            memory.keywords = keywords

        # Save updated memory
        await redis.set(entry_key, json.dumps(memory.to_dict()))
        return True

    async def delete_memory(self, agent_id: str, memory_id: str) -> bool:
        """
        Remove a memory entirely.

        Args:
            agent_id: Agent identifier
            memory_id: Memory to delete

        Returns:
            True if deleted, False if not found
        """
        redis = await self._get_client()

        entry_key = self.ENTRY_KEY.format(agent_id=agent_id, memory_id=memory_id)
        data = await redis.get(entry_key)

        if not data:
            return False

        memory = MemoryEntry.from_dict(json.loads(data))

        # Remove from type set
        if memory.memory_type == MemoryType.SHORT:
            short_key = self.SHORT_SET.format(agent_id=agent_id)
            await redis.srem(short_key, memory_id)
        else:
            long_key = self.LONG_SET.format(agent_id=agent_id)
            await redis.srem(long_key, memory_id)

            # Remove from category
            if memory.category:
                cat_mem_key = self.CATEGORY_MEMORIES.format(
                    agent_id=agent_id, category_id=memory.category
                )
                await redis.srem(cat_mem_key, memory_id)

                # Update category count
                await self._update_category_count(agent_id, memory.category, -1)

        # Remove from importance set
        importance_key = self.BY_IMPORTANCE.format(agent_id=agent_id)
        await redis.zrem(importance_key, memory_id)

        # Remove keyword indexes
        for kw in memory.keywords:
            kw_key = self.KEYWORD_SET.format(agent_id=agent_id, keyword=kw.lower())
            await redis.srem(kw_key, memory_id)

        # Delete the entry
        await redis.delete(entry_key)

        logger.info(f"Deleted memory {memory_id} for {agent_id}")
        return True

    # =========================================================================
    # Short-term Operations
    # =========================================================================

    async def get_short_term_memories(
        self,
        agent_id: str,
        limit: int = SHORT_TERM_MAX_COUNT,
    ) -> List[MemoryEntry]:
        """
        Get all short-term memories for context injection.

        Returns memories sorted by importance (highest first).
        Uses batch fetch to avoid N+1 Redis calls.
        """
        redis = await self._get_client()

        short_key = self.SHORT_SET.format(agent_id=agent_id)
        memory_ids = await redis.smembers(short_key)

        if not memory_ids:
            return []

        # Batch fetch all memories in one mget call
        memories = await self.get_memories_batch(agent_id, list(memory_ids))

        # Sort by importance descending
        memories.sort(key=lambda m: m.importance, reverse=True)
        return memories[:limit]

    async def promote_to_long_term(
        self,
        agent_id: str,
        memory_id: str,
        category: str,
        summary: Optional[str] = None,
    ) -> bool:
        """
        Move memory from short-term to long-term storage.

        Args:
            agent_id: Agent identifier
            memory_id: Memory to promote
            category: Category for organization
            summary: Summary of content (generated if not provided)

        Returns:
            True if promoted, False if not found or already long-term
        """
        redis = await self._get_client()

        entry_key = self.ENTRY_KEY.format(agent_id=agent_id, memory_id=memory_id)
        data = await redis.get(entry_key)

        if not data:
            return False

        memory = MemoryEntry.from_dict(json.loads(data))

        if memory.memory_type == MemoryType.LONG:
            # Already long-term
            return False

        # Update memory
        memory.memory_type = MemoryType.LONG
        memory.category = category
        memory.summary = summary or self._generate_simple_summary(memory.content)

        # Save updated memory
        await redis.set(entry_key, json.dumps(memory.to_dict()))

        # Move between sets
        short_key = self.SHORT_SET.format(agent_id=agent_id)
        long_key = self.LONG_SET.format(agent_id=agent_id)
        await redis.srem(short_key, memory_id)
        await redis.sadd(long_key, memory_id)

        # Ensure category exists
        await self._ensure_category(agent_id, category)

        # Add to category memories
        cat_mem_key = self.CATEGORY_MEMORIES.format(agent_id=agent_id, category_id=category)
        await redis.sadd(cat_mem_key, memory_id)

        # Update category count
        await self._update_category_count(agent_id, category, 1)

        logger.info(f"Promoted memory {memory_id} to long-term category '{category}'")
        return True

    async def _enforce_short_term_limits(self, agent_id: str) -> List[str]:
        """
        Enforce short-term memory limits.

        When limit is exceeded, automatically promotes lowest-importance
        memories to long-term storage.

        Returns IDs of memories that were auto-promoted.
        """
        redis = await self._get_client()

        short_key = self.SHORT_SET.format(agent_id=agent_id)
        count = await redis.scard(short_key)

        if count <= SHORT_TERM_MAX_COUNT:
            return []

        # Get all short-term memories sorted by importance
        memory_ids = await redis.smembers(short_key)
        memories = []
        for memory_id in memory_ids:
            memory = await self.get_memory(agent_id, memory_id)
            if memory:
                memories.append(memory)

        # Sort by importance ascending (lowest first)
        memories.sort(key=lambda m: m.importance)

        # Promote excess memories (lowest importance)
        excess_count = count - SHORT_TERM_MAX_COUNT
        promoted_ids = []

        for memory in memories[:excess_count]:
            # Preserve category if already set (e.g. sentinel's passive_context),
            # otherwise auto-categorize based on content keywords.
            category = memory.category or self._suggest_category(memory.content, memory.keywords)
            await self.promote_to_long_term(
                agent_id=agent_id,
                memory_id=memory.id,
                category=category,
            )
            promoted_ids.append(memory.id)

        if promoted_ids:
            logger.info(f"Auto-promoted {len(promoted_ids)} memories to long-term for {agent_id}")

        return promoted_ids

    # =========================================================================
    # Long-term & Category Operations
    # =========================================================================

    async def get_category_summaries(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get summary of all categories for context.

        Returns list of dicts with name, description, count.
        Uses batch fetch to avoid N+1 Redis calls.
        """
        redis = await self._get_client()

        categories_key = self.CATEGORIES_SET.format(agent_id=agent_id)
        category_ids = await redis.smembers(categories_key)

        if not category_ids:
            return []

        # Batch fetch all categories in one mget call
        keys = [
            self.CATEGORY_KEY.format(agent_id=agent_id, category_id=cat_id)
            for cat_id in category_ids
        ]
        results = await redis.mget(keys)

        summaries = []
        for data in results:
            if data:
                category = MemoryCategory.from_dict(json.loads(data))
                summaries.append({
                    "id": category.id,
                    "name": category.name,
                    "description": category.description,
                    "count": category.memory_count,
                })

        # Sort by memory count descending
        summaries.sort(key=lambda c: c["count"], reverse=True)
        return summaries

    async def get_memories_by_category(
        self,
        agent_id: str,
        category: str,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Retrieve memories from a specific category.

        Uses batch fetch to avoid N+1 Redis calls.
        """
        redis = await self._get_client()

        cat_mem_key = self.CATEGORY_MEMORIES.format(agent_id=agent_id, category_id=category)
        memory_ids = await redis.smembers(cat_mem_key)

        if not memory_ids:
            return []

        # Batch fetch all memories in one mget call
        memories = await self.get_memories_batch(agent_id, list(memory_ids))

        # Sort by importance descending
        memories.sort(key=lambda m: m.importance, reverse=True)
        return memories[:limit]

    async def create_category(
        self,
        agent_id: str,
        category_id: str,
        name: str,
        description: str,
    ) -> MemoryCategory:
        """Create a new memory category."""
        redis = await self._get_client()

        category = MemoryCategory(
            id=category_id,
            agent_id=agent_id,
            name=name,
            description=description,
        )

        cat_key = self.CATEGORY_KEY.format(agent_id=agent_id, category_id=category_id)
        await redis.set(cat_key, json.dumps(category.to_dict()))

        categories_key = self.CATEGORIES_SET.format(agent_id=agent_id)
        await redis.sadd(categories_key, category_id)

        logger.info(f"Created category '{category_id}' for {agent_id}")
        return category

    async def list_categories(self, agent_id: str) -> List[MemoryCategory]:
        """
        List all categories for an agent.

        Uses batch fetch to avoid N+1 Redis calls.
        """
        redis = await self._get_client()

        categories_key = self.CATEGORIES_SET.format(agent_id=agent_id)
        category_ids = await redis.smembers(categories_key)

        if not category_ids:
            return []

        # Batch fetch all categories in one mget call
        keys = [
            self.CATEGORY_KEY.format(agent_id=agent_id, category_id=cat_id)
            for cat_id in category_ids
        ]
        results = await redis.mget(keys)

        categories = []
        for data in results:
            if data:
                categories.append(MemoryCategory.from_dict(json.loads(data)))

        return categories

    async def _ensure_category(self, agent_id: str, category_id: str):
        """Ensure a category exists, creating with defaults if needed."""
        redis = await self._get_client()

        cat_key = self.CATEGORY_KEY.format(agent_id=agent_id, category_id=category_id)
        exists = await redis.exists(cat_key)

        if not exists:
            # Get default description or generate one
            description = DEFAULT_CATEGORIES.get(
                category_id,
                f"Memories categorized as {category_id}"
            )
            await self.create_category(
                agent_id=agent_id,
                category_id=category_id,
                name=category_id.replace("_", " ").title(),
                description=description,
            )

    async def _update_category_count(self, agent_id: str, category_id: str, delta: int):
        """Update memory count for a category."""
        redis = await self._get_client()

        cat_key = self.CATEGORY_KEY.format(agent_id=agent_id, category_id=category_id)
        data = await redis.get(cat_key)

        if data:
            category = MemoryCategory.from_dict(json.loads(data))
            category.memory_count = max(0, category.memory_count + delta)
            category.updated_at = datetime.utcnow()
            await redis.set(cat_key, json.dumps(category.to_dict()))

    # =========================================================================
    # Retrieval & Search
    # =========================================================================

    async def search_memories(
        self,
        agent_id: str,
        query: str,
        memory_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Search memories by keyword matching.

        Args:
            agent_id: Agent identifier
            query: Search query
            memory_type: "short", "long", or None for both
            category: Filter by category (long-term only)
            limit: Maximum results

        Returns:
            List of matching memories sorted by relevance
        """
        redis = await self._get_client()

        # Extract keywords from query
        query_keywords = self._extract_keywords(query)

        # Get candidate memory IDs from keyword indexes
        candidate_ids = set()
        for keyword in query_keywords:
            kw_key = self.KEYWORD_SET.format(agent_id=agent_id, keyword=keyword.lower())
            ids = await redis.smembers(kw_key)
            candidate_ids.update(ids)

        # Also do simple content search for candidates
        if memory_type in (None, "short"):
            short_key = self.SHORT_SET.format(agent_id=agent_id)
            short_ids = await redis.smembers(short_key)
            candidate_ids.update(short_ids)

        if memory_type in (None, "long"):
            if category:
                cat_mem_key = self.CATEGORY_MEMORIES.format(
                    agent_id=agent_id, category_id=category
                )
                cat_ids = await redis.smembers(cat_mem_key)
                candidate_ids.update(cat_ids)
            else:
                long_key = self.LONG_SET.format(agent_id=agent_id)
                long_ids = await redis.smembers(long_key)
                candidate_ids.update(long_ids)

        # Score and rank candidates
        scored_memories = []
        for memory_id in candidate_ids:
            memory = await self.get_memory(agent_id, memory_id)
            if not memory:
                continue

            # Apply type filter
            if memory_type == "short" and memory.memory_type != MemoryType.SHORT:
                continue
            if memory_type == "long" and memory.memory_type != MemoryType.LONG:
                continue

            # Apply category filter
            if category and memory.category != category:
                continue

            # Calculate relevance score
            score = self._calculate_search_score(query, query_keywords, memory)
            if score > 0:
                scored_memories.append((score, memory))

        # Sort by score descending
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_memories[:limit]]

    async def get_relevant_memories(
        self,
        agent_id: str,
        context: str,
        limit: int = AUTO_SELECT_LIMIT,
    ) -> List[MemoryEntry]:
        """
        Get memories most relevant to current context.

        Uses keyword overlap + recency + importance scoring.
        Only searches long-term memories (short-term already included).
        Uses batch fetch to avoid N+1 Redis calls.
        """
        redis = await self._get_client()

        # Extract keywords from context
        context_keywords = self._extract_keywords(context)

        if not context_keywords:
            return []

        # Get all long-term memory IDs
        long_key = self.LONG_SET.format(agent_id=agent_id)
        memory_ids = await redis.smembers(long_key)

        if not memory_ids:
            return []

        # Batch fetch all long-term memories in one mget call
        memories = await self.get_memories_batch(agent_id, list(memory_ids))

        # Score and filter memories
        scored_memories = []
        for memory in memories:
            score = self._calculate_relevance_score(context_keywords, memory)
            if score > 0.1:  # Minimum threshold
                scored_memories.append((score, memory))

        # Sort by score descending
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_memories[:limit]]

    def _calculate_relevance_score(
        self,
        context_keywords: List[str],
        memory: MemoryEntry,
    ) -> float:
        """
        Calculate relevance score for a memory.

        Scoring factors:
        - Keyword overlap: 40%
        - Recency: 20%
        - Importance: 25%
        - Access frequency: 15%
        """
        # Keyword overlap (40%)
        memory_keywords = set(kw.lower() for kw in memory.keywords)
        context_kw_set = set(kw.lower() for kw in context_keywords)
        if memory_keywords and context_kw_set:
            overlap = len(memory_keywords & context_kw_set)
            max_possible = min(len(memory_keywords), len(context_kw_set))
            keyword_score = overlap / max_possible if max_possible > 0 else 0
        else:
            keyword_score = 0

        # Recency (20%) - decay over 30 days
        days_old = (datetime.utcnow() - memory.accessed_at).days
        recency_score = max(0, 1 - (days_old / 30))

        # Importance (25%)
        importance_score = memory.importance

        # Access frequency (15%) - cap at 10 accesses
        frequency_score = min(memory.access_count / 10, 1.0)

        # Weighted combination
        total_score = (
            keyword_score * 0.40 +
            recency_score * 0.20 +
            importance_score * 0.25 +
            frequency_score * 0.15
        )

        return total_score

    def _calculate_search_score(
        self,
        query: str,
        query_keywords: List[str],
        memory: MemoryEntry,
    ) -> float:
        """Calculate search relevance score."""
        score = 0.0

        # Exact phrase match in content
        content_lower = memory.content.lower()
        if query.lower() in content_lower:
            score += 0.5

        # Keyword overlap
        memory_keywords = set(kw.lower() for kw in memory.keywords)
        query_kw_set = set(kw.lower() for kw in query_keywords)
        overlap = len(memory_keywords & query_kw_set)
        if query_kw_set:
            score += 0.3 * (overlap / len(query_kw_set))

        # Importance bonus
        score += 0.2 * memory.importance

        return score

    # =========================================================================
    # Context Building
    # =========================================================================

    async def build_memory_context(
        self,
        agent_id: str,
        current_message: str,
    ) -> MemoryContext:
        """
        Build complete memory context for agent prompt.

        Uses parallel fetches to minimize Redis round-trips.

        Includes:
        - All short-term memories
        - Category headers for long-term
        - Auto-selected relevant long-term memories
        """
        # Fetch all memory data in parallel
        short_term, categories, auto_selected = await asyncio.gather(
            self.get_short_term_memories(agent_id),
            self.get_category_summaries(agent_id),
            self.get_relevant_memories(
                agent_id=agent_id,
                context=current_message,
                limit=AUTO_SELECT_LIMIT,
            ),
        )

        return MemoryContext(
            short_term_memories=short_term,
            category_summaries=categories,
            auto_selected_memories=auto_selected,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract searchable keywords from text."""
        # Simple keyword extraction: words > 3 chars, not common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "what", "which", "who", "whom", "whose", "where",
            "when", "why", "how", "all", "each", "every", "both", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "also", "now", "here",
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Filter and deduplicate
        keywords = []
        seen = set()
        for word in words:
            if len(word) > 3 and word not in stop_words and word not in seen:
                keywords.append(word)
                seen.add(word)

        return keywords[:10]  # Limit to 10 keywords

    def _suggest_category(self, content: str, keywords: List[str]) -> str:
        """Suggest a category based on content and keywords."""
        content_lower = content.lower()

        # Check for category indicators
        if any(w in content_lower for w in ["prefer", "like", "want", "style", "habit"]):
            return "user_preferences"
        if any(w in content_lower for w in ["project", "code", "file", "directory", "repo"]):
            return "project_context"
        if any(w in content_lower for w in ["error", "fix", "solution", "bug", "issue"]):
            return "technical_notes"
        if any(w in content_lower for w in ["workflow", "process", "step", "approach"]):
            return "workflow_patterns"

        # Default category
        return "conversation_highlights"

    def _generate_simple_summary(self, content: str, max_length: int = 200) -> str:
        """Generate a simple summary (truncation with ellipsis)."""
        if len(content) <= max_length:
            return content

        # Truncate at word boundary
        truncated = content[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]

        return truncated + "..."
