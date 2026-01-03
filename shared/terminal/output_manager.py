"""
Live terminal display for Discord.

Displays terminal state in a single Discord message that updates in place.
Designed for CLI tools like Claude Code that use animations and in-place updates.
"""
import asyncio
import time
import logging
from typing import Optional

import discord

logger = logging.getLogger(__name__)


class LiveTerminalDisplay:
    """
    Displays terminal state in a single Discord message.

    Features:
    - Single message that updates in place (like a real terminal)
    - Rate-limited updates (respects Discord's 50 edits/minute limit)
    - Smart truncation when content exceeds Discord's 2000 char limit
    - Skips redundant updates when content hasn't changed
    """

    # Discord rate limit: ~50 edits per minute = 1.2s interval
    MIN_UPDATE_INTERVAL = 1.2
    MAX_CHARS = 1900  # Leave room for code block markers (```)

    def __init__(
        self,
        channel: discord.TextChannel,
        session_id: int,
        update_interval: float = None,
    ):
        self.channel = channel
        self.session_id = session_id
        self.update_interval = update_interval or self.MIN_UPDATE_INTERVAL

        # Message state
        self.message: Optional[discord.Message] = None
        self.current_content: str = ""
        self.pending_content: Optional[str] = None

        # Timing
        self.last_update: float = 0

        # Running state
        self.is_running: bool = True

    async def update(self, content: str) -> None:
        """
        Update display with current terminal content.

        Rate-limited: if called too soon after last update, stores content
        as pending for later flush.

        Args:
            content: Terminal content to display (already cleaned of ANSI codes)
        """
        if not self.is_running:
            return

        # Skip if content unchanged
        if content == self.current_content:
            return

        now = time.time()

        # Rate limit - store pending if too soon
        if now - self.last_update < self.update_interval:
            self.pending_content = content
            return

        await self._do_update(content)

    async def _do_update(self, content: str) -> None:
        """Actually perform the Discord message update."""
        # Truncate to fit Discord limit (keep last N chars)
        display_content = content
        if len(content) > self.MAX_CHARS:
            # Find a good line break point
            truncated = content[-(self.MAX_CHARS - 4):]  # Room for "...\n"
            newline_pos = truncated.find('\n')
            if newline_pos > 0 and newline_pos < 50:
                # Start from the next line for cleaner display
                truncated = truncated[newline_pos + 1:]
            display_content = "...\n" + truncated

        # Format with code block
        formatted = f"```\n{self._escape_codeblock(display_content)}\n```"

        try:
            if self.message:
                await self.message.edit(content=formatted)
            else:
                self.message = await self.channel.send(formatted)

            self.last_update = time.time()
            self.current_content = content
            self.pending_content = None

        except discord.HTTPException as e:
            if e.status == 429:
                # Rate limited - store as pending, will retry later
                logger.debug(f"Session {self.session_id}: Rate limited on update")
                self.pending_content = content
            elif e.code == 10008:
                # Unknown Message - message was deleted
                logger.info(f"Session {self.session_id}: Live message deleted, will recreate")
                self.message = None
                self.pending_content = content
            else:
                logger.error(f"Session {self.session_id}: Failed to update display: {e}")

    async def flush_pending(self) -> None:
        """
        Force update with pending content if enough time has passed.

        Called periodically by the output loop to ensure pending updates
        are eventually displayed.
        """
        if not self.pending_content:
            return

        now = time.time()
        if now - self.last_update >= self.update_interval:
            await self._do_update(self.pending_content)

    def _escape_codeblock(self, text: str) -> str:
        """Escape backticks to prevent breaking code blocks."""
        return text.replace('```', '`\u200b`\u200b`')

    async def cleanup(self) -> None:
        """Clean up when session ends."""
        self.is_running = False

        # Final update with any pending content
        if self.pending_content:
            try:
                await self._do_update(self.pending_content)
            except Exception:
                pass


# Keep old name as alias for backwards compatibility during transition
TerminalOutputManager = LiveTerminalDisplay
