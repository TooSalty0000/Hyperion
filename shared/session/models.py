from dataclasses import dataclass, field
from typing import Optional, Any
import asyncio

from shared.terminal.interface import TerminalBackend
from shared.database.interface import SessionData
from shared.utils import VirtualScreen


@dataclass
class ActiveSession:
    """
    Runtime session combining database record with live components.
    This is the in-memory representation of a running session.
    """
    data: SessionData                              # Persisted metadata
    terminal: TerminalBackend                      # Terminal backend (tmux/pexpect)
    screen: VirtualScreen                          # Terminal emulator for Discord display
    discord_message: Optional[Any] = None          # DEPRECATED: kept for compatibility
    background_task: Optional[asyncio.Task] = None # Output reading task
    display_mode: str = 'scroll'                   # 'scroll' or 'overwrite'
    output_manager: Optional[Any] = None           # TerminalOutputManager for rate-limited output

    @property
    def session_id(self) -> int:
        return self.data.session_id

    @property
    def channel_id(self) -> int:
        return self.data.channel_id

    @property
    def is_alive(self) -> bool:
        return self.terminal is not None and self.terminal.is_alive()

    def get_attach_command(self) -> Optional[str]:
        """Get command for local terminal attachment."""
        if self.terminal:
            return self.terminal.get_attach_command()
        return None
