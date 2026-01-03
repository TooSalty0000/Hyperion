"""Session crash listener for detecting unexpected session terminations."""

import logging
from datetime import datetime
from typing import List, Optional, Set, TYPE_CHECKING

from shared.events.interface import AgentEvent, EventListener, EventPriority

if TYPE_CHECKING:
    from shared.session.registry import SessionRegistry

logger = logging.getLogger(__name__)


class SessionCrashListener(EventListener):
    """
    Monitors tmux sessions for unexpected exits/crashes.

    This listener polls the session registry to detect sessions
    that have died unexpectedly (process terminated without user
    request). When a crash is detected, it emits a high-priority
    event so Altair can respond.
    """

    def __init__(
        self,
        session_registry: 'SessionRegistry',
        poll_interval: float = 5.0,
    ):
        """
        Initialize the crash listener.

        Args:
            session_registry: The SessionRegistry to monitor
            poll_interval: How often to check for crashes (seconds)
        """
        super().__init__("altair.session_crash")
        self._registry = session_registry
        self._poll_interval = poll_interval

        # Track known session IDs to detect disappearances
        self._known_sessions: Set[int] = set()

    @property
    def event_types(self) -> List[str]:
        """Event types this listener can generate."""
        return [
            "session.crashed",
            "session.terminated",
        ]

    async def start(self):
        """Start monitoring for session crashes."""
        # Initialize known sessions
        sessions = await self._registry.list_sessions()
        self._known_sessions = {s.session_id for s in sessions if s.is_alive}

        logger.info(
            f"[{self.listener_id}] Starting with {len(self._known_sessions)} known sessions"
        )

        # Start the polling loop
        import asyncio
        self._task = asyncio.create_task(
            self._run_polling_loop(self._poll_interval)
        )

    async def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        logger.info(f"[{self.listener_id}] Stopped")

    async def _check_for_events(self):
        """
        Check for crashed sessions.

        Compares current alive sessions against known sessions
        to detect unexpected terminations.
        """
        sessions = await self._registry.list_sessions()

        # Get current alive sessions
        current_alive = {s.session_id for s in sessions if s.is_alive}

        # Detect new sessions
        new_sessions = current_alive - self._known_sessions
        for session_id in new_sessions:
            logger.debug(f"[{self.listener_id}] New session detected: {session_id}")

        # Detect crashed sessions (were known, now gone or dead)
        crashed_sessions = self._known_sessions - current_alive

        for crashed_id in crashed_sessions:
            # Find the session data if still in registry
            session = await self._registry.get_session(crashed_id)

            description = f"Session #{crashed_id} has crashed unexpectedly!"
            project_name = None

            if session:
                project_name = session.data.project_name
                if project_name:
                    description = f"Session #{crashed_id} ({project_name}) has crashed unexpectedly!"

            logger.warning(f"[{self.listener_id}] {description}")

            # Emit crash event
            await self.emit_event(AgentEvent(
                event_type="session.crashed",
                source=self.listener_id,
                timestamp=datetime.now(),
                payload={
                    "session_id": crashed_id,
                    "project_name": project_name,
                    "description": description,
                    "suggestion": f"You may want to restart the session or check what went wrong.",
                },
                priority=EventPriority.HIGH,
                event_id=f"session_crash_{crashed_id}_{datetime.now().timestamp()}",
            ))

        # Update known sessions
        self._known_sessions = current_alive

    async def on_session_death(self, session) -> Optional[AgentEvent]:
        """
        Handle session death callback from SessionRegistry.

        This can be used as an alternative to polling - the registry
        can call this directly when it detects a dead session.

        Args:
            session: The ActiveSession that died

        Returns:
            AgentEvent if one should be emitted
        """
        description = f"Session #{session.session_id} process died!"
        project_name = session.data.project_name

        if project_name:
            description = f"Session #{session.session_id} ({project_name}) process died!"

        event = AgentEvent(
            event_type="session.crashed",
            source=self.listener_id,
            timestamp=datetime.now(),
            payload={
                "session_id": session.session_id,
                "project_name": project_name,
                "channel_id": session.channel_id,
                "description": description,
                "suggestion": "You may want to restart the session or check what went wrong.",
            },
            priority=EventPriority.HIGH,
            event_id=f"session_crash_{session.session_id}_{datetime.now().timestamp()}",
        )

        # Emit directly if we have a callback
        await self.emit_event(event)

        # Remove from known sessions
        self._known_sessions.discard(session.session_id)

        return event
