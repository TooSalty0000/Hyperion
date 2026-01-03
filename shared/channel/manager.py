import discord
from discord.ext import commands
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ChannelManager:
    """
    Manages Discord channel creation and deletion for CLI sessions.
    """

    # Channel naming pattern
    CHANNEL_PREFIX = "vega-cli-"
    CATEGORY_NAME = "Vega CLI Sessions"

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._category_cache: dict = {}  # guild_id -> category

    async def _get_or_create_category(self, guild: discord.Guild) -> Optional[discord.CategoryChannel]:
        """Get or create the Vega category for organizing channels."""
        if guild.id in self._category_cache:
            # Verify it still exists
            category = self._category_cache[guild.id]
            if category in guild.categories:
                return category
            else:
                del self._category_cache[guild.id]

        # Find existing category
        for category in guild.categories:
            if category.name == self.CATEGORY_NAME:
                self._category_cache[guild.id] = category
                return category

        # Create new category
        try:
            category = await guild.create_category(
                self.CATEGORY_NAME,
                reason="Vega CLI session management"
            )
            self._category_cache[guild.id] = category
            logger.info(f"Created category '{self.CATEGORY_NAME}' in guild {guild.name}")
            return category
        except discord.Forbidden:
            logger.error(f"No permission to create category in guild {guild.name}")
            return None
        except discord.HTTPException as e:
            logger.error(f"Failed to create category: {e}")
            return None

    async def create_channel(self, guild: discord.Guild,
                             session_id: int) -> Optional[discord.TextChannel]:
        """
        Create a new text channel for a CLI session.
        Returns the created channel or None on failure.
        """
        channel_name = f"{self.CHANNEL_PREFIX}{session_id}"

        # Get category (optional, graceful fallback)
        category = await self._get_or_create_category(guild)

        try:
            channel = await guild.create_text_channel(
                channel_name,
                category=category,
                topic=f"Vega CLI Session #{session_id}",
                reason=f"Vega CLI session {session_id} created"
            )
            logger.info(f"Created channel #{channel_name} (ID: {channel.id})")
            return channel
        except discord.Forbidden:
            logger.error(f"No permission to create channel in guild {guild.name}")
            return None
        except discord.HTTPException as e:
            logger.error(f"Failed to create channel: {e}")
            return None

    async def delete_channel(self, channel_id: int) -> bool:
        """
        Delete a CLI session channel.
        Returns success.
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            logger.warning(f"Channel {channel_id} not found for deletion")
            return False

        try:
            channel_name = channel.name
            await channel.delete(reason="Vega CLI session ended")
            logger.info(f"Deleted channel {channel_name} (ID: {channel_id})")
            return True
        except discord.Forbidden:
            logger.error(f"No permission to delete channel {channel_id}")
            return False
        except discord.HTTPException as e:
            logger.error(f"Failed to delete channel {channel_id}: {e}")
            return False

    def is_vega_channel(self, channel: discord.abc.GuildChannel) -> bool:
        """Check if a channel is a Vega CLI session channel."""
        if not hasattr(channel, 'name'):
            return False
        return channel.name.startswith(self.CHANNEL_PREFIX)

    def extract_session_id(self, channel: discord.abc.GuildChannel) -> Optional[int]:
        """Extract session ID from channel name."""
        if not self.is_vega_channel(channel):
            return None
        try:
            return int(channel.name[len(self.CHANNEL_PREFIX):])
        except ValueError:
            return None
