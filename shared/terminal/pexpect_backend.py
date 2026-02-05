import asyncio
import pexpect
import logging
from typing import Optional

from .interface import TerminalBackend, TerminalConfig

logger = logging.getLogger(__name__)


class PexpectBackend(TerminalBackend):
    """
    Fallback pexpect-based terminal backend.

    This is the original implementation - runs process directly via pexpect.
    No local window attachment available.
    """

    def __init__(self, config: TerminalConfig):
        super().__init__(config)
        self.process: Optional[pexpect.spawn] = None
        self._reader_task: Optional[asyncio.Task] = None

    @staticmethod
    def is_available() -> bool:
        """Pexpect is always available (it's a dependency)."""
        return True

    @staticmethod
    def get_name() -> str:
        return "pexpect"

    async def start(self) -> bool:
        """Start the process via pexpect."""
        try:
            self.process = pexpect.spawn(
                self.config.command,
                cwd=self.config.cwd,
                encoding='utf-8',
                echo=False,
                dimensions=(self.config.rows, self.config.cols)
            )

            # Start background reader
            self._reader_task = asyncio.create_task(self._read_loop())

            logger.info(f"Started pexpect process: {self.config.command}")
            return True

        except Exception as e:
            logger.error(f"Error starting pexpect process: {e}")
            return False

    async def _read_loop(self):
        """Continuously read output from process."""
        while self.process and self.process.isalive():
            try:
                chunk = await asyncio.to_thread(
                    self.process.read_nonblocking, 1024, timeout=0.1
                )
                if chunk:
                    print(chunk, end='', flush=True)  # Echo locally
                    await self._output_queue.put(chunk)

            except pexpect.TIMEOUT:
                await asyncio.sleep(0.1)
            except pexpect.EOF:
                logger.info("Process finished (EOF)")
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading output: {e}")
                break

    async def send_input(self, text: str) -> bool:
        """Send text to process."""
        if not self.process or not self.process.isalive():
            return False

        try:
            if not text.endswith('\n'):
                text += '\n'
            self.process.send(text)
            return True
        except Exception as e:
            logger.error(f"Error sending input: {e}")
            return False

    async def send_control(self, key: str) -> bool:
        """Send control character."""
        if not self.process or not self.process.isalive():
            return False

        try:
            special_keys = {
                "up": "\x1b[A",
                "down": "\x1b[B",
                "right": "\x1b[C",
                "left": "\x1b[D",
                "enter": "\r",
                "esc": "\x1b",
                "escape": "\x1b",
                "backspace": "\x08",
                "tab": "\t"
            }

            key_lower = key.lower()
            if key_lower in special_keys:
                self.process.send(special_keys[key_lower])
            else:
                # Assume it's a control character like 'c' for Ctrl+C
                self.process.sendcontrol(key_lower)

            return True
        except Exception as e:
            logger.error(f"Error sending control: {e}")
            return False

    async def get_output(self) -> Optional[str]:
        """Get output from queue."""
        try:
            return await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    async def resize(self, cols: int, rows: int) -> bool:
        """Resize the terminal."""
        if not self.process:
            return False

        try:
            self.process.setwinsize(rows, cols)
            self.config.cols = cols
            self.config.rows = rows
            return True
        except Exception as e:
            logger.error(f"Error resizing: {e}")
            return False

    async def kill(self) -> bool:
        """Kill the process."""
        try:
            if self._reader_task:
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass

            if self.process and self.process.isalive():
                self.process.terminate(force=True)
                self.process.close()

            self.process = None
            return True

        except Exception as e:
            logger.error(f"Error killing process: {e}")
            return False

    def is_alive(self) -> bool:
        """Check if process is alive."""
        return self.process is not None and self.process.isalive()

    def get_attach_command(self) -> Optional[str]:
        """No local attachment available for pexpect."""
        return None

    def get_pid(self) -> Optional[int]:
        """Get process ID."""
        if self.process:
            return self.process.pid
        return None

    def get_foreground_process(self) -> Optional[str]:
        """Not available for pexpect backend."""
        return None
