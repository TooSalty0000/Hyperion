from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import asyncio


@dataclass
class TerminalConfig:
    """Configuration for terminal backend."""
    cols: int = 80          # Terminal width
    rows: int = 24          # Terminal height
    command: str = "zsh"    # Command to run
    cwd: Optional[str] = None  # Working directory
    session_name: Optional[str] = None  # Session identifier


class TerminalBackend(ABC):
    """Abstract interface for terminal backends."""

    def __init__(self, config: TerminalConfig):
        self.config = config
        self._output_queue: asyncio.Queue = asyncio.Queue()

    @property
    def output_queue(self) -> asyncio.Queue:
        return self._output_queue

    @abstractmethod
    async def start(self) -> bool:
        """Start the terminal session. Returns success."""
        pass

    @abstractmethod
    async def send_input(self, text: str) -> bool:
        """Send text input to the terminal. Returns success."""
        pass

    @abstractmethod
    async def send_control(self, key: str) -> bool:
        """Send control character (e.g., 'c' for Ctrl+C). Returns success."""
        pass

    @abstractmethod
    async def get_output(self) -> Optional[str]:
        """Get current terminal output/screen content."""
        pass

    @abstractmethod
    async def resize(self, cols: int, rows: int) -> bool:
        """Resize the terminal. Returns success."""
        pass

    @abstractmethod
    async def kill(self) -> bool:
        """Kill the terminal session. Returns success."""
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if the session is still running."""
        pass

    @abstractmethod
    def get_attach_command(self) -> Optional[str]:
        """Get command for local user to attach to this session."""
        pass

    @abstractmethod
    def get_pid(self) -> Optional[int]:
        """Get the process ID if available."""
        pass

    @abstractmethod
    def get_foreground_process(self) -> Optional[str]:
        """Get the name of the foreground process (e.g. 'bash', 'claude')."""
        pass

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if this backend is available on the current system."""
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Get the backend name."""
        pass
