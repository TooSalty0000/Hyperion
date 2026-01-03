import asyncio
import subprocess
import shutil
import logging
import time
from typing import Optional

from .interface import TerminalBackend, TerminalConfig

logger = logging.getLogger(__name__)


class TmuxBackend(TerminalBackend):
    """
    Tmux-based terminal backend.

    Creates a tmux session that:
    - Has controllable size (cols x rows)
    - Can be attached locally via: tmux attach -t <session_name>
    - Receives input via tmux send-keys
    - Output captured via tmux capture-pane
    """

    def __init__(self, config: TerminalConfig):
        super().__init__(config)
        self.session_name = config.session_name or f"vega-{id(self)}"
        self._alive = False
        self._reader_task: Optional[asyncio.Task] = None
        self._last_output = ""

    @staticmethod
    def is_available() -> bool:
        """Check if tmux is installed."""
        return shutil.which("tmux") is not None

    @staticmethod
    def get_name() -> str:
        return "tmux"

    async def start(self) -> bool:
        """Create a new tmux session."""
        try:
            # Kill any existing session with same name
            await self._run_tmux("kill-session", "-t", self.session_name, check=False)

            # Create new detached session with specified size
            result = await self._run_tmux(
                "new-session",
                "-d",  # Detached
                "-s", self.session_name,
                "-x", str(self.config.cols),
                "-y", str(self.config.rows),
                "-c", self.config.cwd or ".",
                self.config.command
            )

            if result.returncode != 0:
                logger.error(f"Failed to create tmux session: {result.stderr}")
                return False

            self._alive = True

            # Start background reader
            self._reader_task = asyncio.create_task(self._read_loop())

            logger.info(f"Started tmux session: {self.session_name} ({self.config.cols}x{self.config.rows})")
            return True

        except Exception as e:
            logger.error(f"Error starting tmux session: {e}")
            return False

    async def attach_existing(self) -> bool:
        """Attach to an existing tmux session without creating a new one.

        Used for session recovery after bot restart.
        """
        try:
            # Check if session exists
            result = await self._run_tmux("has-session", "-t", self.session_name, check=False)
            if result.returncode != 0:
                logger.error(f"Tmux session {self.session_name} does not exist")
                return False

            self._alive = True

            # Start background reader
            self._reader_task = asyncio.create_task(self._read_loop())

            logger.info(f"Attached to existing tmux session: {self.session_name}")
            return True

        except Exception as e:
            logger.error(f"Error attaching to tmux session: {e}")
            return False

    async def _run_tmux(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        """Run a tmux command."""
        cmd = ["tmux"] + list(args)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        return subprocess.CompletedProcess(
            cmd, proc.returncode,
            stdout.decode('utf-8', errors='replace'),
            stderr.decode('utf-8', errors='replace')
        )

    async def _capture_pane_raw(self) -> Optional[str]:
        """Capture current pane content (internal use for read loop).

        Captures only the visible pane content, not scrollback history.
        This is ideal for live display of CLI tools with animations.
        """
        try:
            result = await self._run_tmux(
                "capture-pane",
                "-t", self.session_name,
                "-p",  # Print to stdout
                "-J",  # Join wrapped lines
                # No -S flag = only visible content (not scrollback)
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception as e:
            logger.error(f"Error capturing pane: {e}")
            return None

    async def capture_full_history(self) -> Optional[str]:
        """Capture full pane content including scrollback history.

        Use this for scrollback commands, not for live display.
        """
        try:
            result = await self._run_tmux(
                "capture-pane",
                "-t", self.session_name,
                "-p",  # Print to stdout
                "-J",  # Join wrapped lines
                "-S", "-"  # From start of history
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception as e:
            logger.error(f"Error capturing full history: {e}")
            return None

    async def _read_loop(self):
        """Continuously read output from tmux and push to queue."""
        while self._alive:
            try:
                output = await self._capture_pane_raw()
                if output is not None and output != self._last_output:
                    # Always push full output to queue for consumers
                    await self._output_queue.put(output)
                    self._last_output = output

                # Poll more frequently for responsive updates
                await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in tmux read loop: {e}")
                await asyncio.sleep(0.5)

    async def send_input(self, text: str) -> bool:
        """Send text to the tmux session."""
        try:
            # Check if text ends with newline (means we should press Enter after)
            should_press_enter = text.endswith('\n')
            if should_press_enter:
                text = text.rstrip('\n')

            # Send text with literal flag to preserve special characters
            if text:
                result = await self._run_tmux(
                    "send-keys",
                    "-t", self.session_name,
                    "-l",  # Literal - don't interpret special keys
                    text
                )
                if result.returncode != 0:
                    return False

            # Send Enter key separately if requested
            if should_press_enter:
                result = await self._run_tmux(
                    "send-keys",
                    "-t", self.session_name,
                    "Enter"  # Without -l, this sends the Enter key
                )
                return result.returncode == 0

            return True

        except Exception as e:
            logger.error(f"Error sending input: {e}")
            return False

    async def send_control(self, key: str) -> bool:
        """Send control character to tmux session."""
        try:
            # Map key names to tmux key names
            key_map = {
                'c': 'C-c',      # Ctrl+C
                'd': 'C-d',      # Ctrl+D
                'z': 'C-z',      # Ctrl+Z
                'l': 'C-l',      # Ctrl+L (clear)
                'up': 'Up',
                'down': 'Down',
                'left': 'Left',
                'right': 'Right',
                'enter': 'Enter',
                'esc': 'Escape',
                'escape': 'Escape',
                'tab': 'Tab',
                'backspace': 'BSpace',
            }

            tmux_key = key_map.get(key.lower(), key)

            result = await self._run_tmux(
                "send-keys",
                "-t", self.session_name,
                tmux_key
            )
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error sending control key: {e}")
            return False

    async def get_output(self) -> Optional[str]:
        """Capture full pane content including scrollback.

        This is the public API used by scroll commands and history requests.
        """
        return await self.capture_full_history()

    async def resize(self, cols: int, rows: int) -> bool:
        """Resize the tmux pane."""
        try:
            # Resize the window
            result = await self._run_tmux(
                "resize-window",
                "-t", self.session_name,
                "-x", str(cols),
                "-y", str(rows)
            )

            if result.returncode == 0:
                self.config.cols = cols
                self.config.rows = rows
                logger.info(f"Resized tmux session to {cols}x{rows}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error resizing: {e}")
            return False

    async def kill(self) -> bool:
        """Kill the tmux session."""
        try:
            self._alive = False

            if self._reader_task:
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass

            result = await self._run_tmux("kill-session", "-t", self.session_name, check=False)
            logger.info(f"Killed tmux session: {self.session_name}")
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Error killing session: {e}")
            return False

    def is_alive(self) -> bool:
        """Check if tmux session exists."""
        if not self._alive:
            return False

        # Synchronous check using subprocess
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", self.session_name],
                capture_output=True
            )
            alive = result.returncode == 0
            if not alive:
                self._alive = False
            return alive
        except Exception:
            self._alive = False
            return False

    def get_attach_command(self) -> Optional[str]:
        """Get command for local attachment."""
        return f"tmux attach -t {self.session_name}"

    def get_pid(self) -> Optional[int]:
        """Get the PID of the process in tmux."""
        try:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", self.session_name, "-F", "#{pane_pid}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split('\n')[0])
        except Exception:
            pass
        return None
