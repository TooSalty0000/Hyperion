"""Message queue system for sequential agent processing with state awareness."""

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Awaitable, Any, List, Dict, Tuple, TYPE_CHECKING

from enum import Enum

import discord

if TYPE_CHECKING:
    from shared.llm.interface import LLMProvider

logger = logging.getLogger(__name__)


class QueueItemStatus(Enum):
    """Status of a queued message."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskRelationship(Enum):
    """How a new message relates to the current task."""
    UNRELATED = "unrelated"  # Different task entirely
    CONTINUATION = "continuation"  # Adds to or follows up on current work
    DUPLICATE = "duplicate"  # Same request repeated
    CORRECTION = "correction"  # Changes or redirects current work
    UNKNOWN = "unknown"  # Needs LLM to evaluate


@dataclass
class TaskState:
    """
    Rich state tracking for current agent task.

    Provides context that can be shared with other agents or
    used by the LLM to understand ongoing work.
    """
    content: str  # The task description/request
    started_at: datetime
    source_agent: Optional[str] = None  # If from another agent
    source_user_id: Optional[int] = None
    channel_id: Optional[int] = None

    # Dynamic state updated during processing
    status_updates: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> int:
        """Seconds since task started."""
        return int((datetime.now() - self.started_at).total_seconds())

    def add_status(self, status: str):
        """Add a status update."""
        self.status_updates.append(f"[{self.elapsed_seconds}s] {status}")

    def summary(self) -> str:
        """Get a human-readable summary of current task state."""
        parts = [f"Task: {self.content[:100]}"]
        parts.append(f"Running for: {self.elapsed_seconds}s")
        if self.tools_used:
            parts.append(f"Tools used: {', '.join(self.tools_used[-3:])}")  # Last 3
        if self.status_updates:
            parts.append(f"Latest: {self.status_updates[-1]}")
        return " | ".join(parts)


@dataclass
class QueuedMessage:
    """A message waiting to be processed."""
    message: discord.Message
    context: Any  # AgentContext
    is_from_agent: bool
    queued_at: datetime = field(default_factory=datetime.now)
    status: QueueItemStatus = QueueItemStatus.PENDING
    position: int = 0  # Position in queue when added
    message_id: int = 0  # Discord message ID for deduplication
    # Context about agent state when this was queued
    task_context_when_queued: Optional[str] = None


@dataclass
class RecentTask:
    """
    Track recently processed/in-progress tasks for content-based deduplication.

    When multiple agents request the same task (e.g., "find X"), this allows
    the receiving agent to detect duplicates and respond only once.
    """
    content_hash: str  # SHA256 of normalized content for fast matching
    content_preview: str  # First 100 chars for debugging
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[str] = None  # Cached result for sharing with late requesters
    requesting_agents: List[str] = field(default_factory=list)  # Agents who requested this
    requesting_message_ids: List[int] = field(default_factory=list)  # Their message IDs

    @property
    def is_complete(self) -> bool:
        """Check if task has completed."""
        return self.completed_at is not None

    @property
    def age_seconds(self) -> float:
        """Seconds since task started."""
        return (datetime.now() - self.started_at).total_seconds()


class AgentMessageQueue:
    """
    Sequential message queue for an agent with state awareness.

    Key design principles:
    - Messages are always queued (never blocked based on content)
    - Rich task state is tracked and available for context
    - The LLM decides how to handle related/duplicate requests
    - Agents can check for new messages during long operations
    """

    def __init__(
        self,
        agent_name: str,
        process_callback: Callable[[discord.Message, Any, bool], Awaitable[None]],
        max_queue_size: int = 50,
        utility_llm: Optional['LLMProvider'] = None,
    ):
        """
        Initialize the message queue.

        Args:
            agent_name: Name of the agent (for logging/display)
            process_callback: Async function to process messages
                              Signature: (message, context, is_from_agent) -> None
            max_queue_size: Maximum queued messages before rejecting
            utility_llm: Optional cheap/fast LLM for similarity detection
        """
        self.agent_name = agent_name
        self._process_callback = process_callback
        self._max_queue_size = max_queue_size
        self._utility_llm = utility_llm

        self._queue: asyncio.Queue[QueuedMessage] = asyncio.Queue()
        self._is_processing = False
        self._current_item: Optional[QueuedMessage] = None
        self._processor_task: Optional[asyncio.Task] = None
        self._total_processed = 0
        self._total_queued = 0

        # Track recently processed message IDs to prevent exact duplicates
        self._processed_message_ids: set[int] = set()
        self._max_tracked_ids = 100

        # Rich task state tracking
        self._current_task: Optional[TaskState] = None
        self._current_task_hash: Optional[str] = None  # Hash of current task for result caching

        # Content-based deduplication for multi-agent requests
        self._recent_tasks: Dict[str, RecentTask] = {}  # hash -> RecentTask
        self._task_ttl_seconds: int = 60  # How long to remember tasks

        # Optional callback for task state changes
        self._on_task_change: Optional[Callable[[Optional[TaskState]], Awaitable[None]]] = None

    def start(self):
        """Start the queue processor."""
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._process_loop())
            logger.info(f"{self.agent_name} message queue started")

    def stop(self):
        """Stop the queue processor."""
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            logger.info(f"{self.agent_name} message queue stopped")

    def set_task_change_callback(self, callback: Callable[[Optional[TaskState]], Awaitable[None]]):
        """Set callback for task state changes (start/end)."""
        self._on_task_change = callback

    @property
    def is_busy(self) -> bool:
        """Check if agent is currently processing a message."""
        return self._is_processing

    @property
    def queue_size(self) -> int:
        """Number of messages waiting in queue."""
        return self._queue.qsize()

    @property
    def current_task(self) -> Optional[TaskState]:
        """Get current task state (if any)."""
        return self._current_task

    @property
    def current_task_info(self) -> Optional[str]:
        """Get info about current task being processed."""
        if self._current_task:
            return self._current_task.summary()
        return None

    @property
    def current_task_content(self) -> Optional[str]:
        """Get the content of the current task."""
        if self._current_task:
            return self._current_task.content
        return None

    def update_task_status(self, status: str):
        """Update the current task's status (call from agent during processing)."""
        if self._current_task:
            self._current_task.add_status(status)

    def record_tool_use(self, tool_name: str):
        """Record that a tool was used (call from agent during processing)."""
        if self._current_task:
            self._current_task.tools_used.append(tool_name)

    # =========================================================================
    # Content-Based Deduplication
    # =========================================================================

    def _normalize_content(self, content: str) -> str:
        """
        Normalize content for hashing - remove variable parts.

        Strips:
        - Whitespace variations
        - Source agent markers like [from: vega]
        - Common prefixes like [TASK], [STATUS], etc.
        """
        # Remove source markers
        content = re.sub(r'\[from:\s*\w+\]', '', content, flags=re.IGNORECASE)
        # Remove task/status prefixes
        content = re.sub(r'\[(TASK|STATUS|ACTION|HANDOFF)[^\]]*\]', '', content, flags=re.IGNORECASE)
        # Normalize whitespace
        content = ' '.join(content.split())
        # Lowercase for comparison
        return content.lower().strip()

    def _hash_content(self, content: str) -> str:
        """Generate SHA256 hash of normalized content."""
        normalized = self._normalize_content(content)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]  # Short hash

    def _extract_source_agent(self, message: discord.Message) -> Optional[str]:
        """Extract source agent name from message content or author."""
        # Check for [from: agent_name] marker
        match = re.search(r'\[from:\s*(\w+)\]', message.content, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        # Fall back to author name
        return message.author.name.lower() if message.author else None

    def _cleanup_old_tasks(self):
        """Remove tasks older than TTL."""
        now = datetime.now()
        expired = [
            h for h, t in self._recent_tasks.items()
            if t.age_seconds > self._task_ttl_seconds
        ]
        for h in expired:
            del self._recent_tasks[h]
            logger.debug(f"[{self.agent_name}] Cleaned up expired task: {h}")

    async def _check_semantic_similarity(
        self,
        new_content: str,
        existing_content: str,
    ) -> TaskRelationship:
        """
        Use LLM to check if two requests are semantically similar.

        Falls back to UNKNOWN if no LLM is available.
        """
        if not self._utility_llm:
            return TaskRelationship.UNKNOWN

        try:
            from shared.llm.types import Message, Role

            prompt = f"""Compare these two requests to an AI agent. Are they asking for the SAME thing?

Request 1: {existing_content[:200]}
Request 2: {new_content[:200]}

Reply with ONLY one word:
- DUPLICATE: Same request, just different wording
- CONTINUATION: Request 2 adds to or follows up on Request 1
- CORRECTION: Request 2 changes or corrects Request 1
- UNRELATED: Completely different requests"""

            response = await self._utility_llm.complete(
                messages=[Message(role=Role.USER, content=prompt)],
                max_tokens=20,
            )

            result = response.content.strip().upper()
            if result in ["DUPLICATE", "CONTINUATION", "CORRECTION", "UNRELATED"]:
                return TaskRelationship(result.lower())

        except Exception as e:
            logger.warning(f"[{self.agent_name}] Semantic similarity check failed: {e}")

        return TaskRelationship.UNKNOWN

    def mark_task_complete(self, result: str):
        """
        Mark the current task as complete and cache the result.

        Call this from the cog after processing completes to enable
        result sharing with duplicate requests.
        """
        if self._current_task_hash and self._current_task_hash in self._recent_tasks:
            task = self._recent_tasks[self._current_task_hash]
            task.completed_at = datetime.now()
            task.result = result[:1000]  # Limit cached result size

            if len(task.requesting_agents) > 1:
                logger.info(
                    f"[{self.agent_name}] Task completed - also requested by: "
                    f"{', '.join(task.requesting_agents[1:])}"
                )

    async def enqueue(
        self,
        message: discord.Message,
        context: Any,
        is_from_agent: bool,
    ) -> Tuple[bool, str, Optional[RecentTask]]:
        """
        Add a message to the queue with content-based deduplication.

        Messages are rejected if:
        - Exact duplicate message ID
        - Content-based duplicate from another agent (within TTL)
        - Queue is full

        Args:
            message: Discord message
            context: AgentContext for processing
            is_from_agent: Whether message is from another agent

        Returns:
            Tuple of (success, status, recent_task)
            - success: Whether message was enqueued
            - status: Status string explaining the result
              - "ok" if enqueued successfully
              - "duplicate_id" for exact message ID duplicate
              - "duplicate_task" for content-based duplicate
              - "queue_full" if queue is at capacity
              - "busy:<summary>" if enqueued while processing
              - "queued:<position>" if enqueued behind others
            - recent_task: RecentTask if duplicate, None otherwise
        """
        msg_id = message.id

        # Cleanup old tasks periodically
        self._cleanup_old_tasks()

        # Check for exact duplicate message ID
        if msg_id in self._processed_message_ids:
            logger.warning(
                f"[{self.agent_name}] DUPLICATE MSG ID: {msg_id} already seen, skipping. "
                f"Content: {message.content[:50]}..."
            )
            return False, "duplicate_id", None

        # Content-based deduplication for inter-agent requests
        if is_from_agent:
            content_hash = self._hash_content(context.message_content)
            source_agent = self._extract_source_agent(message)

            # Check if same content is already being processed or was recently processed
            if content_hash in self._recent_tasks:
                recent = self._recent_tasks[content_hash]

                if recent.age_seconds < self._task_ttl_seconds:
                    # This is a duplicate request - add to the list of requesters
                    if source_agent and source_agent not in recent.requesting_agents:
                        recent.requesting_agents.append(source_agent)
                        recent.requesting_message_ids.append(msg_id)

                    logger.info(
                        f"[{self.agent_name}] DUPLICATE TASK detected from {source_agent}: "
                        f"'{recent.content_preview[:50]}...' (hash={content_hash})"
                    )
                    return False, "duplicate_task", recent

            # Check current task for semantic similarity (LLM fallback)
            if self._current_task and self._utility_llm:
                relationship = await self._check_semantic_similarity(
                    context.message_content,
                    self._current_task.content,
                )
                if relationship == TaskRelationship.DUPLICATE:
                    # Create a RecentTask for the current task if not exists
                    current_hash = self._hash_content(self._current_task.content)
                    if current_hash not in self._recent_tasks:
                        self._recent_tasks[current_hash] = RecentTask(
                            content_hash=current_hash,
                            content_preview=self._current_task.content[:100],
                            started_at=self._current_task.started_at,
                            requesting_agents=[self._current_task.source_agent] if self._current_task.source_agent else [],
                            requesting_message_ids=[],
                        )
                    recent = self._recent_tasks[current_hash]
                    if source_agent and source_agent not in recent.requesting_agents:
                        recent.requesting_agents.append(source_agent)
                        recent.requesting_message_ids.append(msg_id)

                    logger.info(
                        f"[{self.agent_name}] SEMANTIC DUPLICATE detected from {source_agent}: "
                        f"LLM determined '{context.message_content[:30]}' duplicates current task"
                    )
                    return False, "duplicate_task", recent

            # New task from agent - track it
            self._recent_tasks[content_hash] = RecentTask(
                content_hash=content_hash,
                content_preview=context.message_content[:100],
                started_at=datetime.now(),
                requesting_agents=[source_agent] if source_agent else [],
                requesting_message_ids=[msg_id],
            )

        # Check queue capacity
        if self._queue.qsize() >= self._max_queue_size:
            return False, "queue_full", None

        self._total_queued += 1
        position = self._queue.qsize() + 1

        # Capture current task context when queueing
        task_context = None
        if self._current_task:
            task_context = self._current_task.summary()

        item = QueuedMessage(
            message=message,
            context=context,
            is_from_agent=is_from_agent,
            position=position,
            message_id=msg_id,
            task_context_when_queued=task_context,
        )

        # Track this message ID
        self._processed_message_ids.add(msg_id)
        if len(self._processed_message_ids) > self._max_tracked_ids:
            oldest = next(iter(self._processed_message_ids))
            self._processed_message_ids.discard(oldest)

        # Track content hash for this task
        if is_from_agent:
            self._current_task_hash = self._hash_content(context.message_content)

        logger.info(
            f"[{self.agent_name}] ENQUEUE: msg_id={msg_id}, from_agent={is_from_agent}, "
            f"position={position}, busy={self._is_processing}, content={message.content[:50]}..."
        )

        await self._queue.put(item)

        # Return appropriate status
        if not self._is_processing:
            return True, "ok", None  # Will process immediately

        # Currently busy - provide rich context
        if self._current_task:
            return True, f"busy:{self._current_task.summary()}", None
        else:
            if position == 1:
                return True, "queued:1", None
            return True, f"queued:{position}", None

    async def _process_loop(self):
        """Background loop that processes queued messages."""
        logger.info(f"{self.agent_name} queue processor started")

        while True:
            try:
                # Wait for next item
                item = await self._queue.get()

                # Calculate queue wait time
                queue_wait_ms = (datetime.now() - item.queued_at).total_seconds() * 1000

                self._is_processing = True
                self._current_item = item

                # Create rich task state
                source_agent = None
                if item.is_from_agent:
                    # Try to extract source agent name from message
                    source_agent = getattr(item.message.author, 'name', None)

                self._current_task = TaskState(
                    content=item.context.message_content,
                    started_at=datetime.now(),
                    source_agent=source_agent,
                    source_user_id=item.context.user_id,
                    channel_id=item.context.channel_id,
                )

                item.status = QueueItemStatus.PROCESSING

                logger.info(
                    f"[{self.agent_name}] PROCESS START: msg_id={item.message_id}, "
                    f"queue_remaining={self._queue.qsize()}, content={item.context.message_content[:50]}..."
                )
                logger.debug(
                    f"[Queue] {self.agent_name}: queue_wait={queue_wait_ms:.0f}ms, "
                    f"from_agent={item.is_from_agent}"
                )

                # Notify task change if callback set
                if self._on_task_change:
                    try:
                        await self._on_task_change(self._current_task)
                    except Exception as e:
                        logger.warning(f"Task change callback error: {e}")

                try:
                    # Process the message
                    await self._process_callback(
                        item.message,
                        item.context,
                        item.is_from_agent,
                    )
                    item.status = QueueItemStatus.COMPLETED
                    self._total_processed += 1
                    logger.info(
                        f"[{self.agent_name}] PROCESS COMPLETE: msg_id={item.message_id}, "
                        f"total_processed={self._total_processed}"
                    )

                except Exception as e:
                    item.status = QueueItemStatus.FAILED
                    logger.error(f"[{self.agent_name}] PROCESS FAILED: msg_id={item.message_id}, error={e}")

                    # Try to notify the channel
                    try:
                        await item.message.channel.send(
                            f"Sorry, I encountered an error processing your message: {str(e)[:200]}"
                        )
                    except Exception:
                        pass

                finally:
                    self._current_item = None
                    self._current_task = None
                    self._is_processing = False
                    self._queue.task_done()

                    # Notify task ended
                    if self._on_task_change:
                        try:
                            await self._on_task_change(None)
                        except Exception as e:
                            logger.warning(f"Task change callback error: {e}")

            except asyncio.CancelledError:
                logger.info(f"{self.agent_name} queue processor cancelled")
                break
            except Exception as e:
                logger.error(f"{self.agent_name} queue processor error: {e}")
                self._is_processing = False
                await asyncio.sleep(1)

    def get_status(self) -> dict:
        """Get queue status information."""
        return {
            "agent": self.agent_name,
            "is_busy": self._is_processing,
            "queue_size": self._queue.qsize(),
            "current_task": self._current_task.summary() if self._current_task else None,
            "current_task_content": self._current_task.content if self._current_task else None,
            "total_processed": self._total_processed,
            "total_queued": self._total_queued,
        }

    def get_state_for_llm(self) -> Optional[str]:
        """
        Get current state formatted for LLM context injection.

        Returns a string that can be injected into the agent's context
        to help it understand if a new message relates to ongoing work.
        """
        if not self._current_task:
            return None

        lines = [
            f"[AGENT STATE: Currently processing a task]",
            f"Task: {self._current_task.content}",
            f"Running for: {self._current_task.elapsed_seconds} seconds",
        ]

        if self._current_task.tools_used:
            lines.append(f"Tools used so far: {', '.join(self._current_task.tools_used[-5:])}")

        if self._current_task.status_updates:
            lines.append(f"Recent progress: {self._current_task.status_updates[-1]}")

        pending = self._queue.qsize()
        if pending > 0:
            lines.append(f"Messages waiting in queue: {pending}")

        return "\n".join(lines)
