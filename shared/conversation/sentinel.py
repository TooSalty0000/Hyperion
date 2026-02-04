"""Context Sentinel - passive chat monitoring and summarization.

Observes all Discord messages (even those not directed at us), buffers them
in Redis, and produces summaries when the buffer reaches a configurable
threshold.  Summaries are persisted both in the memory system (under the
``passive_context`` category, owned by Vega) and as plain-text files on disk.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import discord
    from shared.llm.interface import LLMProvider
    from shared.memory.manager import MemoryManager

logger = logging.getLogger(__name__)

# Lazy Redis import (matches codebase convention)
_redis = None


def _import_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError(
                "redis is required for ContextSentinel. "
                "Install it with: pip install redis"
            )
    return _redis


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The agent_id used when storing memories — Vega owns all passive context.
OWNER_AGENT_ID = "vega"

DEFAULT_BUFFER_SIZE = 50
FLUSH_TIMEOUT_SECONDS = 300  # Hard cap on summarization wall-clock time
SUMMARY_PERSIST_DIR = os.path.expanduser("~/.hyperion/vega/passive_context")

# Redis key patterns
BUFFER_KEY = "hyperion:sentinel:{channel_id}:buffer"
SUMMARY_COUNT_KEY = "hyperion:sentinel:{channel_id}:summary_count"


class ContextSentinel:
    """
    Passive chat observer that buffers messages and produces periodic
    summaries for Vega's ambient awareness.

    Lifecycle::

        observe(message)  ──►  buffer in Redis
                                │
                                ▼  (buffer_size reached)
                           _flush_buffer()  [async, up to 300 s]
                                │
                           ┌────┴────┐
                           ▼         ▼
                   Memory system   File on disk
                  (agent: vega)    (~/.hyperion/…)

    All stored memories use ``agent_id="vega"`` and
    ``category="passive_context"`` so they are retrievable via Vega's
    normal memory tools.
    """

    def __init__(
        self,
        redis_url: str,
        memory_manager: Optional['MemoryManager'] = None,
        utility_llm: Optional['LLMProvider'] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ):
        self._redis_url = redis_url
        self._redis = None
        self._memory_manager = memory_manager
        self._utility_llm = utility_llm
        self._buffer_size = buffer_size
        self._flush_lock = asyncio.Lock()

        # Track running flush tasks so we can cancel on shutdown
        self._active_flush_tasks: Dict[int, asyncio.Task] = {}

        # Ensure persistence directory exists
        Path(SUMMARY_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    async def _get_client(self):
        if self._redis is None:
            redis_async = _import_redis()
            self._redis = await redis_async.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def observe(
        self,
        message: 'discord.Message',
        agent_registry: Dict[str, int],
    ) -> None:
        """
        Record a message into the channel buffer.

        Call this for **every** message the bot can see, regardless of
        whether it is directed at Vega.  The sentinel decides internally
        when to flush and summarize.

        Args:
            message: The raw Discord message.
            agent_registry: ``{agent_name: user_id}`` mapping for author
                identification.
        """
        if not self._redis_url:
            return

        try:
            channel_id = message.channel.id
            entry = self._serialize_message(message, agent_registry)

            redis = await self._get_client()
            buf_key = BUFFER_KEY.format(channel_id=channel_id)

            await redis.rpush(buf_key, json.dumps(entry))
            buf_len = await redis.llen(buf_key)

            if buf_len >= self._buffer_size:
                # Skip if a flush is already running for this channel
                existing = self._active_flush_tasks.get(channel_id)
                if existing and not existing.done():
                    logger.debug(
                        f"[Sentinel] Flush already in progress for ch {channel_id}"
                    )
                    return

                task = asyncio.create_task(
                    self._guarded_flush(channel_id),
                    name=f"sentinel-flush-{channel_id}",
                )
                self._active_flush_tasks[channel_id] = task

        except Exception as exc:
            # Never let sentinel errors disrupt the main message flow
            logger.warning(f"[Sentinel] observe error: {exc}")

    async def get_recent_summary(self, channel_id: int) -> Optional[str]:
        """
        Return the most recent passive-context summary for *channel_id*.

        Returns ``None`` if no summaries exist yet.
        """
        path = self._summary_path(channel_id)
        if not path.exists():
            return None

        try:
            text = path.read_text(encoding="utf-8")
            sections = text.strip().split("\n---\n")
            return sections[-1].strip() if sections else None
        except Exception as exc:
            logger.warning(f"[Sentinel] Failed to read summary file: {exc}")
            return None

    async def get_all_summaries(self, channel_id: int) -> Optional[str]:
        """Return all accumulated summaries for *channel_id*."""
        path = self._summary_path(channel_id)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except Exception as exc:
            logger.warning(f"[Sentinel] Failed to read summary file: {exc}")
            return None

    async def recall_passive_context(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Any]:
        """
        Search Vega's passive-context memories by keyword relevance.

        This is the hook for Vega to pull ambient context into her
        active working set — call it when building the system prompt or
        when the context window is getting full and you want to swap in
        the most relevant background knowledge.

        Args:
            query: A natural-language query or the current conversation
                topic to match against.
            limit: Maximum number of memory entries to return.

        Returns:
            A list of ``MemoryEntry`` objects from the ``passive_context``
            category, sorted by relevance.  Empty list if the memory
            system is unavailable.
        """
        if not self._memory_manager:
            return []

        try:
            return await self._memory_manager.search_memories(
                agent_id=OWNER_AGENT_ID,
                query=query,
                category="passive_context",
                limit=limit,
            )
        except Exception as exc:
            logger.warning(f"[Sentinel] recall_passive_context error: {exc}")
            return []

    async def shutdown(self) -> None:
        """Cancel any in-flight flush tasks (call on cog_unload)."""
        for channel_id, task in self._active_flush_tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"[Sentinel] Cancelled flush for ch {channel_id}")
        self._active_flush_tasks.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _guarded_flush(self, channel_id: int) -> None:
        """Wrap ``_flush_buffer`` with a hard timeout."""
        try:
            await asyncio.wait_for(
                self._flush_buffer(channel_id),
                timeout=FLUSH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[Sentinel] Flush for ch {channel_id} timed out "
                f"after {FLUSH_TIMEOUT_SECONDS}s"
            )
        except asyncio.CancelledError:
            logger.info(f"[Sentinel] Flush for ch {channel_id} cancelled")
        except Exception as exc:
            logger.error(f"[Sentinel] guarded_flush error: {exc}", exc_info=True)
        finally:
            self._active_flush_tasks.pop(channel_id, None)

    async def _flush_buffer(self, channel_id: int) -> None:
        """Drain the buffer, summarize, and persist the result."""
        async with self._flush_lock:
            redis = await self._get_client()
            buf_key = BUFFER_KEY.format(channel_id=channel_id)

            # Atomically drain the buffer
            raw_entries: List[str] = []
            while True:
                entry = await redis.lpop(buf_key)
                if entry is None:
                    break
                raw_entries.append(entry)

            if not raw_entries:
                return

            messages = [json.loads(e) for e in raw_entries]
            logger.info(
                f"[Sentinel] Flushing {len(messages)} messages "
                f"from channel {channel_id}"
            )

            summary = await self._summarize_messages(messages)
            await self._store_summary(channel_id, summary)

            # Bump the summary counter
            count_key = SUMMARY_COUNT_KEY.format(channel_id=channel_id)
            await redis.incr(count_key)

    def _serialize_message(
        self,
        message: 'discord.Message',
        agent_registry: Dict[str, int],
    ) -> Dict[str, Any]:
        """Turn a Discord message into a lightweight dict for the buffer."""
        author = message.author.display_name
        for name, uid in agent_registry.items():
            if message.author.id == uid:
                author = f"[agent:{name}]"
                break

        return {
            "author": author,
            "content": (message.content or "")[:500],
            "channel": getattr(message.channel, "name", str(message.channel.id)),
            "ts": (
                message.created_at.isoformat()
                if message.created_at
                else datetime.now(timezone.utc).isoformat()
            ),
        }

    async def _summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Produce a concise summary of the buffered messages."""
        transcript_lines = []
        for m in messages:
            ts_short = m["ts"][:19] if "ts" in m else ""
            transcript_lines.append(
                f"[{ts_short}] {m['author']}: {m['content']}"
            )
        transcript = "\n".join(transcript_lines)

        if self._utility_llm:
            try:
                return await self._llm_summarize(transcript)
            except Exception as exc:
                logger.warning(
                    f"[Sentinel] LLM summarize failed, using fallback: {exc}"
                )

        return self._fallback_summarize(messages)

    async def _llm_summarize(self, transcript: str) -> str:
        """Use the utility LLM to produce a structured summary."""
        from shared.llm.types import Message, Role

        llm_messages = [
            Message(
                role=Role.SYSTEM,
                content=(
                    "You are a concise note-taker. Given a transcript of Discord "
                    "chat messages, produce a short summary (200 words max) that "
                    "captures:\n"
                    "1. Key topics discussed\n"
                    "2. Decisions made or actions requested\n"
                    "3. Notable participants and their roles\n"
                    "4. Any unresolved questions or open threads\n\n"
                    "Output ONLY the summary — no preamble, no labels."
                ),
            ),
            Message(role=Role.USER, content=transcript),
        ]

        response = await self._utility_llm.complete(messages=llm_messages)
        return response.content.strip()

    def _fallback_summarize(self, messages: List[Dict[str, Any]]) -> str:
        """Simple extractive fallback when no LLM is available."""
        authors = {m["author"] for m in messages}
        first_ts = messages[0].get("ts", "?")[:19]
        last_ts = messages[-1].get("ts", "?")[:19]

        sample_lines = []
        for m in messages[:3]:
            sample_lines.append(f"  {m['author']}: {m['content'][:100]}")
        if len(messages) > 6:
            sample_lines.append(f"  ... ({len(messages) - 6} more messages) ...")
        for m in messages[-3:]:
            sample_lines.append(f"  {m['author']}: {m['content'][:100]}")

        sample = "\n".join(sample_lines)

        return (
            f"Chat summary ({first_ts} to {last_ts})\n"
            f"Participants: {', '.join(sorted(authors))}\n"
            f"Message count: {len(messages)}\n"
            f"Sample:\n{sample}"
        )

    async def _store_summary(self, channel_id: int, summary: str) -> None:
        """Persist the summary to the memory system and to a file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header = f"[{timestamp} | channel:{channel_id}]"
        full_entry = f"{header}\n{summary}"

        # 1. Append to persistent file
        try:
            path = self._summary_path(channel_id)
            with open(path, "a", encoding="utf-8") as f:
                f.write(full_entry + "\n---\n")
            logger.info(f"[Sentinel] Summary written to {path}")
        except Exception as exc:
            logger.error(f"[Sentinel] File persist failed: {exc}")

        # 2. Store in memory system under Vega's ownership
        if self._memory_manager:
            try:
                await self._memory_manager.store_memory(
                    agent_id=OWNER_AGENT_ID,
                    content=full_entry,
                    importance=0.4,
                    category="passive_context",
                    keywords=self._extract_keywords(summary),
                )
                logger.info("[Sentinel] Summary stored in memory system")
            except Exception as exc:
                logger.error(f"[Sentinel] Memory store failed: {exc}")

    def _summary_path(self, channel_id: int) -> Path:
        """Return the file path for a channel's summaries."""
        return Path(SUMMARY_PERSIST_DIR) / f"channel_{channel_id}.txt"

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract a handful of keywords for memory indexing."""
        import re

        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are",
            "were", "been", "be", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must",
            "this", "that", "these", "those", "not", "all", "some", "any",
            "about", "also", "just", "than", "then", "them", "they", "what",
            "which", "who", "how", "when", "where", "chat", "summary",
        }
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        seen = set()
        keywords = []
        for w in words:
            if w not in stop_words and w not in seen:
                keywords.append(w)
                seen.add(w)
            if len(keywords) >= 8:
                break
        return keywords
