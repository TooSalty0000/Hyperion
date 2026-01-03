"""Abstract event listener interface for agent event handling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority level for events."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class AgentEvent:
    """
    An event that can trigger agent action.

    Contains all context needed for an agent to respond to an
    external event (calendar reminder, session crash, etc).
    """
    event_type: str  # e.g., "calendar.reminder", "session.crashed"
    source: str  # Listener ID that generated this event
    timestamp: datetime = field(default_factory=datetime.now)

    # Event-specific data
    payload: Dict[str, Any] = field(default_factory=dict)

    # Where to send the response
    target_channel_id: Optional[int] = None

    # Processing hints
    priority: EventPriority = EventPriority.NORMAL

    # Deduplication - set to prevent duplicate processing
    event_id: Optional[str] = None

    def get_description(self) -> str:
        """Get human-readable description of this event."""
        return self.payload.get("description", str(self.payload))

    def format_for_discord(self) -> str:
        """Format event for Discord message."""
        priority_emoji = {
            EventPriority.LOW: "ℹ️",
            EventPriority.NORMAL: "📢",
            EventPriority.HIGH: "⚠️",
            EventPriority.CRITICAL: "🚨",
        }
        emoji = priority_emoji.get(self.priority, "📢")
        return f"{emoji} **{self.event_type}**: {self.get_description()}"


class EventListener(ABC):
    """
    Abstract base class for event listeners.

    Event listeners monitor external sources (calendars, sessions, files,
    webhooks, etc.) and emit AgentEvents when something noteworthy occurs.

    Subclasses must implement:
    - event_types: List of event types this listener can emit
    - start(): Begin listening
    - stop(): Stop listening
    - _check_for_events(): Check for new events (called by polling loop)

    Example implementations:
    - CalendarEventListener: Polls Google Calendar for upcoming events
    - SessionCrashListener: Monitors tmux sessions for crashes
    - WebhookListener: Receives HTTP webhooks from external services
    """

    def __init__(self, listener_id: str):
        """
        Initialize the listener.

        Args:
            listener_id: Unique identifier for this listener (e.g., "polaris.calendar")
        """
        self.listener_id = listener_id
        self._running = False
        self._event_callback: Optional[Callable[[AgentEvent], Awaitable[None]]] = None
        self._task: Optional[asyncio.Task] = None

    @property
    @abstractmethod
    def event_types(self) -> List[str]:
        """
        List of event types this listener can generate.

        Used for subscription filtering.
        Examples: ["calendar.reminder", "calendar.starting_soon"]
        """
        pass

    @property
    def is_running(self) -> bool:
        """Whether the listener is currently active."""
        return self._running

    def set_event_callback(
        self,
        callback: Callable[[AgentEvent], Awaitable[None]]
    ):
        """
        Set the callback for when events are detected.

        The EventDispatcher calls this to register itself.
        """
        self._event_callback = callback

    async def emit_event(self, event: AgentEvent):
        """
        Emit an event to the registered callback.

        Called by subclasses when they detect an event.
        """
        if self._event_callback:
            try:
                await self._event_callback(event)
                logger.debug(f"[{self.listener_id}] Emitted event: {event.event_type}")
            except Exception as e:
                logger.error(f"[{self.listener_id}] Error emitting event: {e}")

    @abstractmethod
    async def start(self):
        """
        Start listening for events.

        Implementation should:
        1. Initialize any connections or state
        2. Start background task(s) that monitor for events
        3. Call emit_event() when events are detected
        """
        pass

    @abstractmethod
    async def stop(self):
        """
        Stop listening for events.

        Implementation should:
        1. Set _running = False
        2. Cancel any background tasks
        3. Clean up connections or resources
        """
        pass

    async def _run_polling_loop(self, interval: float = 5.0):
        """
        Helper method for polling-based listeners.

        Runs a loop that calls _check_for_events() at regular intervals.
        Subclasses can use this in their start() method.

        Args:
            interval: Seconds between polls (default 5.0)
        """
        self._running = True
        logger.info(f"[{self.listener_id}] Starting polling loop (interval={interval}s)")

        while self._running:
            try:
                await self._check_for_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.listener_id}] Error in polling loop: {e}")

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

        self._running = False
        logger.info(f"[{self.listener_id}] Polling loop stopped")

    @abstractmethod
    async def _check_for_events(self):
        """
        Check for new events.

        Called by the polling loop. Implementation should:
        1. Check the external source for new events
        2. Compare against previous state to detect changes
        3. Call emit_event() for any new events
        """
        pass


class WebhookEventListener(EventListener):
    """
    Base class for webhook-based event listeners.

    Use this for listeners that receive push notifications via HTTP
    rather than polling. Subclasses implement handle_webhook() to
    process incoming payloads.
    """

    @abstractmethod
    async def handle_webhook(
        self,
        payload: Dict[str, Any]
    ) -> Optional[AgentEvent]:
        """
        Handle an incoming webhook payload.

        Args:
            payload: The webhook payload data

        Returns:
            AgentEvent if the webhook should trigger agent action,
            None otherwise.
        """
        pass

    async def start(self):
        """Webhook listeners don't need a polling loop."""
        self._running = True
        logger.info(f"[{self.listener_id}] Webhook listener ready")

    async def stop(self):
        """Stop the webhook listener."""
        self._running = False
        logger.info(f"[{self.listener_id}] Webhook listener stopped")

    async def _check_for_events(self):
        """Not used for webhook listeners."""
        pass
