"""Session registry for managing terminal sessions."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Awaitable

from shared.terminal import TerminalConfig, create_terminal_backend
from shared.database.interface import SessionStore, SessionStatus
from shared.utils import VirtualScreen
from .models import ActiveSession

logger = logging.getLogger(__name__)


def _process_name_to_mode(process_name: Optional[str]) -> Optional[str]:
    """Map a foreground process name to a session mode.

    Returns "claude_code", "shell", or None (unknown — don't change mode).
    """
    if not process_name:
        return None
    name = process_name.lower().strip()
    if "claude" in name:
        return "claude_code"
    if name in ("bash", "zsh", "sh", "fish", "dash", "ksh", "tcsh", "csh"):
        return "shell"
    return None  # Unknown process — don't change mode


@dataclass
class SessionDefaults:
    """Default configuration for session creation."""
    cli_command: str = "bash"
    workspace_dir: str = "."
    terminal_cols: int = 120
    terminal_rows: int = 40
    terminal_backend: str = "auto"


class SessionRegistry:
    """
    Manages all active CLI sessions.
    Bridges database persistence with runtime terminal instances.
    """

    def __init__(self, store: SessionStore, defaults: SessionDefaults = None):
        self._store = store
        self._defaults = defaults or SessionDefaults()
        self._sessions: Dict[int, ActiveSession] = {}  # session_id -> ActiveSession
        self._channel_map: Dict[int, int] = {}         # channel_id -> session_id

        # Callback for process death notification
        self._on_process_death: Optional[Callable[[ActiveSession], Awaitable[None]]] = None

    def set_death_callback(self, callback: Callable[[ActiveSession], Awaitable[None]]):
        """Set callback to invoke when a process terminates."""
        self._on_process_death = callback

    async def create_session(self, channel_id: int,
                             command: str = None,
                             workspace_dir: str = None,
                             cols: int = None,
                             rows: int = None,
                             project_name: str = None) -> ActiveSession:
        """Create and start a new session for the given channel."""
        command = command or self._defaults.cli_command
        workspace_dir = workspace_dir or self._defaults.workspace_dir
        cols = cols or self._defaults.terminal_cols
        rows = rows or self._defaults.terminal_rows

        # Get next session ID for naming
        next_id = await self._store.get_next_id()

        # Persist to store
        session_data = await self._store.create(
            channel_id=channel_id,
            command=command,
            workspace_dir=workspace_dir,
            project_name=project_name
        )

        # Create terminal config
        terminal_config = TerminalConfig(
            cols=cols,
            rows=rows,
            command=command,
            cwd=workspace_dir,
            session_name=f"vega-{session_data.session_id}"
        )

        # Create terminal backend
        terminal = create_terminal_backend(
            terminal_config,
            preferred_backend=self._defaults.terminal_backend
        )

        # Create screen emulator
        screen = VirtualScreen()

        active_session = ActiveSession(
            data=session_data,
            terminal=terminal,
            screen=screen
        )

        # Start the terminal
        success = await terminal.start()
        if not success:
            await self._store.delete(session_data.session_id)
            raise RuntimeError(f"Failed to start terminal backend")

        # Update with PID
        pid = terminal.get_pid()
        await self._store.update(
            session_data.session_id,
            pid=pid,
            status=SessionStatus.RUNNING
        )
        active_session.data.pid = pid
        active_session.data.status = SessionStatus.RUNNING

        # Register
        self._sessions[session_data.session_id] = active_session
        self._channel_map[channel_id] = session_data.session_id

        logger.info(f"Created session {session_data.session_id} for channel {channel_id} "
                    f"using {terminal.get_name()} backend ({cols}x{rows})")
        return active_session

    async def get_session(self, session_id: int) -> Optional[ActiveSession]:
        """Get active session by ID."""
        return self._sessions.get(session_id)

    async def get_session_by_channel(self, channel_id: int) -> Optional[ActiveSession]:
        """Get active session for a Discord channel."""
        session_id = self._channel_map.get(channel_id)
        if session_id:
            return self._sessions.get(session_id)
        return None

    async def get_session_by_project(self, project_name: str) -> Optional[ActiveSession]:
        """Get active session for a project by name."""
        for session in self._sessions.values():
            if session.data.project_name == project_name:
                return session
        return None

    async def get_stored_session_by_project(self, project_name: str):
        """Get stored session data by project name (may not be active)."""
        return await self._store.get_by_project(project_name)

    async def terminate_session(self, session_id: int) -> bool:
        """
        Terminate a session by ID. Returns success.
        Does NOT delete the channel - that's handled by ChannelManager.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False

        # Cancel background task if running
        if session.background_task:
            session.background_task.cancel()
            try:
                await session.background_task
            except asyncio.CancelledError:
                pass

        # Kill terminal
        await session.terminal.kill()

        # Update store
        await self._store.update(session_id, status=SessionStatus.STOPPED)

        # Cleanup maps
        channel_id = session.channel_id
        del self._sessions[session_id]
        if channel_id in self._channel_map:
            del self._channel_map[channel_id]

        # Remove from persistent store
        await self._store.delete(session_id)

        logger.info(f"Terminated session {session_id}")
        return True

    async def terminate_by_channel(self, channel_id: int) -> bool:
        """Terminate session by channel ID."""
        session_id = self._channel_map.get(channel_id)
        if session_id:
            return await self.terminate_session(session_id)
        return False

    async def list_sessions(self) -> List[ActiveSession]:
        """List all active sessions."""
        return list(self._sessions.values())

    async def check_health(self):
        """
        Health check - detect dead processes and trigger cleanup.
        Called periodically by background task.
        """
        dead_sessions = []

        for session_id, session in list(self._sessions.items()):
            if not session.is_alive:
                dead_sessions.append(session)

        for session in dead_sessions:
            logger.warning(f"Session {session.session_id} process died")
            if self._on_process_death:
                await self._on_process_death(session)

    async def shutdown_all(self):
        """Gracefully shutdown all sessions."""
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.terminate_session(session_id)

    async def get_next_id(self) -> int:
        """Get next available session ID."""
        return await self._store.get_next_id()

    async def update_mode(self, session_id: int, mode: str) -> bool:
        """Update the mode ('shell' or 'claude_code') for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        old_mode = getattr(session.data, 'mode', 'shell')
        if old_mode == mode:
            return True
        session.data.mode = mode
        await self._store.update(session_id, mode=mode)
        logger.info(f"Session {session_id} mode changed: {old_mode} -> {mode}")
        return True

    async def recover_sessions(self, start_output_loop_callback=None) -> List[ActiveSession]:
        """
        Recover sessions from persistent store on startup.

        Checks if tmux sessions are still alive and reconnects to them.
        Dead sessions are cleaned up.

        Args:
            start_output_loop_callback: Optional callback to start output loops for recovered sessions

        Returns:
            List of recovered ActiveSession objects
        """
        import subprocess

        stored_sessions = await self._store.list_all()
        recovered = []

        for session_data in stored_sessions:
            if session_data.status != SessionStatus.RUNNING:
                # Skip non-running sessions
                await self._store.delete(session_data.session_id)
                continue

            session_name = f"vega-{session_data.session_id}"

            # Check if tmux session exists
            try:
                result = subprocess.run(
                    ["tmux", "has-session", "-t", session_name],
                    capture_output=True,
                    timeout=5
                )
                tmux_alive = result.returncode == 0
            except Exception:
                tmux_alive = False

            if not tmux_alive:
                # Session is dead, clean up
                logger.info(f"Session {session_data.session_id} tmux not found, cleaning up")
                await self._store.delete(session_data.session_id)
                continue

            # Tmux session is alive, reconnect to it
            logger.info(f"Recovering session {session_data.session_id} (tmux: {session_name})")

            try:
                # Create terminal config
                terminal_config = TerminalConfig(
                    cols=self._defaults.terminal_cols,
                    rows=self._defaults.terminal_rows,
                    command=session_data.command or self._defaults.cli_command,
                    cwd=session_data.workspace_dir or self._defaults.workspace_dir,
                    session_name=session_name
                )

                # Create terminal backend in attach mode
                terminal = create_terminal_backend(
                    terminal_config,
                    preferred_backend="tmux"
                )

                # Attach to existing session instead of starting new one
                if hasattr(terminal, 'attach_existing'):
                    success = await terminal.attach_existing()
                else:
                    # Fallback: try to start (tmux backend should detect existing session)
                    success = await terminal.start()

                if not success:
                    logger.warning(f"Failed to attach to session {session_data.session_id}")
                    await self._store.delete(session_data.session_id)
                    continue

                # Create screen emulator
                screen = VirtualScreen()

                active_session = ActiveSession(
                    data=session_data,
                    terminal=terminal,
                    screen=screen
                )

                # Register
                self._sessions[session_data.session_id] = active_session
                if session_data.channel_id:
                    self._channel_map[session_data.channel_id] = session_data.session_id

                # Reconcile mode with live foreground process
                detected = _process_name_to_mode(terminal.get_foreground_process())
                if detected and detected != getattr(session_data, 'mode', 'shell'):
                    session_data.mode = detected
                    await self._store.update(session_data.session_id, mode=detected)
                    logger.info(f"Session {session_data.session_id} mode reconciled to {detected}")

                # Start output loop if callback provided
                if start_output_loop_callback:
                    start_output_loop_callback(active_session)

                recovered.append(active_session)
                logger.info(f"Successfully recovered session {session_data.session_id}")

            except Exception as e:
                logger.error(f"Failed to recover session {session_data.session_id}: {e}")
                await self._store.delete(session_data.session_id)

        logger.info(f"Recovered {len(recovered)} sessions from persistent store")
        return recovered
