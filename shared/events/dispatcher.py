"""Event dispatcher for routing events to agents."""

import logging
from typing import Awaitable, Callable, Dict, List, Optional, Set

from shared.events.interface import AgentEvent, EventListener

logger = logging.getLogger(__name__)


class EventDispatcher:
    """
    Central dispatcher for routing events to agents.

    The dispatcher:
    1. Manages event listener lifecycle (start/stop)
    2. Routes events to subscribed agents
    3. Handles event deduplication
    4. Sends events to main channel so all agents can hear

    Usage:
        dispatcher = EventDispatcher(main_channel_id=123456)

        # Register listeners
        dispatcher.register_listener(session_crash_listener)
        dispatcher.register_listener(calendar_listener)

        # Subscribe agents to event types
        dispatcher.subscribe_agent("altair", ["session.*"], altair_handler)
        dispatcher.subscribe_agent("polaris", ["calendar.*"], polaris_handler)

        # Start all listeners
        await dispatcher.start_all()
    """

    def __init__(
        self,
        main_channel_id: int,
        discord_bot: Optional[object] = None,
    ):
        """
        Initialize the dispatcher.

        Args:
            main_channel_id: Discord channel ID for event notifications
            discord_bot: Optional Discord bot for sending messages
        """
        self._main_channel_id = main_channel_id
        self._discord_bot = discord_bot

        # Registered listeners
        self._listeners: Dict[str, EventListener] = {}

        # Agent subscriptions: agent_name -> set of event type patterns
        self._subscriptions: Dict[str, Set[str]] = {}

        # Agent handlers: agent_name -> async handler function
        self._handlers: Dict[str, Callable[[AgentEvent], Awaitable[None]]] = {}

        # Processed event IDs for deduplication
        self._processed_events: Set[str] = set()
        self._max_processed_cache = 1000

    @property
    def main_channel_id(self) -> int:
        """Get the main channel ID."""
        return self._main_channel_id

    def set_discord_bot(self, bot: object):
        """Set the Discord bot for sending messages."""
        self._discord_bot = bot

    def register_listener(self, listener: EventListener):
        """
        Register an event listener.

        The listener will have its callback set to route events
        through this dispatcher.

        Args:
            listener: The EventListener to register
        """
        listener.set_event_callback(self._on_event)
        self._listeners[listener.listener_id] = listener
        logger.info(
            f"Registered event listener: {listener.listener_id} "
            f"(types: {listener.event_types})"
        )

    def unregister_listener(self, listener_id: str):
        """
        Unregister an event listener.

        Args:
            listener_id: ID of the listener to remove
        """
        if listener_id in self._listeners:
            del self._listeners[listener_id]
            logger.info(f"Unregistered event listener: {listener_id}")

    def subscribe_agent(
        self,
        agent_name: str,
        event_types: List[str],
        handler: Callable[[AgentEvent], Awaitable[None]],
    ):
        """
        Subscribe an agent to specific event types.

        Args:
            agent_name: Name of the agent subscribing
            event_types: List of event type patterns to subscribe to.
                         Supports wildcards like "calendar.*"
            handler: Async function to call when matching events occur
        """
        self._subscriptions[agent_name] = set(event_types)
        self._handlers[agent_name] = handler
        logger.info(f"Agent '{agent_name}' subscribed to: {event_types}")

    def unsubscribe_agent(self, agent_name: str):
        """
        Unsubscribe an agent from all events.

        Args:
            agent_name: Name of the agent to unsubscribe
        """
        self._subscriptions.pop(agent_name, None)
        self._handlers.pop(agent_name, None)
        logger.info(f"Agent '{agent_name}' unsubscribed from all events")

    async def start_all(self):
        """Start all registered listeners."""
        logger.info(f"Starting {len(self._listeners)} event listeners")
        for listener in self._listeners.values():
            try:
                await listener.start()
            except Exception as e:
                logger.error(f"Failed to start listener {listener.listener_id}: {e}")

    async def stop_all(self):
        """Stop all registered listeners."""
        logger.info(f"Stopping {len(self._listeners)} event listeners")
        for listener in self._listeners.values():
            try:
                await listener.stop()
            except Exception as e:
                logger.error(f"Failed to stop listener {listener.listener_id}: {e}")

    async def _on_event(self, event: AgentEvent):
        """
        Handle incoming event from a listener.

        Routes the event to subscribed agents and optionally
        sends to the main Discord channel.
        """
        # Deduplication check
        if event.event_id:
            if event.event_id in self._processed_events:
                logger.debug(f"Skipping duplicate event: {event.event_id}")
                return
            self._processed_events.add(event.event_id)
            self._trim_processed_cache()

        # Set target channel if not specified
        if event.target_channel_id is None:
            event.target_channel_id = self._main_channel_id

        logger.info(
            f"Dispatching event: {event.event_type} from {event.source} "
            f"(priority={event.priority.name})"
        )

        # Send to Discord main channel so all agents can see
        await self._send_to_discord(event)

        # Route to subscribed agents
        for agent_name, subscribed_types in self._subscriptions.items():
            if self._matches_subscription(event.event_type, subscribed_types):
                handler = self._handlers.get(agent_name)
                if handler:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(
                            f"Error in event handler for {agent_name}: {e}"
                        )

    async def _send_to_discord(self, event: AgentEvent):
        """Send event notification to Discord channel."""
        if not self._discord_bot:
            return

        channel = self._discord_bot.get_channel(event.target_channel_id)
        if not channel:
            logger.warning(
                f"Could not find channel {event.target_channel_id} for event"
            )
            return

        try:
            message = event.format_for_discord()
            await channel.send(message, silent=True)
        except Exception as e:
            logger.error(f"Failed to send event to Discord: {e}")

    def _matches_subscription(
        self,
        event_type: str,
        subscriptions: Set[str]
    ) -> bool:
        """
        Check if event type matches any subscription pattern.

        Supports wildcard patterns like "calendar.*" which matches
        "calendar.reminder", "calendar.starting_soon", etc.
        """
        for pattern in subscriptions:
            if pattern == event_type:
                return True
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if event_type.startswith(prefix + "."):
                    return True
            if pattern == "*":
                return True
        return False

    def _trim_processed_cache(self):
        """Trim the processed events cache if it gets too large."""
        if len(self._processed_events) > self._max_processed_cache:
            # Keep the most recent half
            excess = len(self._processed_events) - (self._max_processed_cache // 2)
            for _ in range(excess):
                self._processed_events.pop()

    def get_listener(self, listener_id: str) -> Optional[EventListener]:
        """Get a registered listener by ID."""
        return self._listeners.get(listener_id)

    def list_listeners(self) -> List[str]:
        """Get list of registered listener IDs."""
        return list(self._listeners.keys())

    def list_subscriptions(self) -> Dict[str, Set[str]]:
        """Get all agent subscriptions."""
        return dict(self._subscriptions)
