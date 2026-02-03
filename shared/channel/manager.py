import re

import discord
from discord.ext import commands
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ChannelManager:
    """
    Manages Discord channel creation and deletion for CLI sessions.
    """

    # Channel naming patterns
    CHANNEL_PREFIX = "vega-cli-"
    CATEGORY_NAME = "Vega CLI Sessions"

    PROJECT_CHANNEL_PREFIX = "project-"
    DEPARTMENT_CHANNEL_PREFIX = "dept-"
    PROJECT_CATEGORY_NAME = "Hyperion Projects"

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._category_cache: dict = {}  # guild_id -> category
        self._project_category_cache: dict = {}  # guild_id -> project category

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

    # --------------------------------------------------
    # PROJECT CHANNELS (Meeting Rooms)
    # --------------------------------------------------

    async def _get_or_create_project_category(
        self, guild: discord.Guild
    ) -> Optional[discord.CategoryChannel]:
        """Get or create the Hyperion Projects category."""
        if guild.id in self._project_category_cache:
            category = self._project_category_cache[guild.id]
            if category in guild.categories:
                return category
            else:
                del self._project_category_cache[guild.id]

        for category in guild.categories:
            if category.name == self.PROJECT_CATEGORY_NAME:
                self._project_category_cache[guild.id] = category
                return category

        try:
            category = await guild.create_category(
                self.PROJECT_CATEGORY_NAME,
                reason="Hyperion project meeting rooms",
            )
            self._project_category_cache[guild.id] = category
            logger.info(
                f"Created category '{self.PROJECT_CATEGORY_NAME}' in guild {guild.name}"
            )
            return category
        except discord.Forbidden:
            logger.error(
                f"No permission to create project category in guild {guild.name}"
            )
            return None
        except discord.HTTPException as e:
            logger.error(f"Failed to create project category: {e}")
            return None

    async def create_project_channel(
        self,
        guild: discord.Guild,
        graph_id: str,
        goal: str,
    ) -> Optional[discord.TextChannel]:
        """
        Create a meeting room channel for a job graph.

        Returns the created channel or None on failure.
        """
        # Build channel name: project-{graph_id}-{sanitized_goal}
        sanitized = re.sub(r"[^a-z0-9-]", "", goal.lower().replace(" ", "-"))[:30]
        channel_name = f"{self.PROJECT_CHANNEL_PREFIX}{graph_id}"
        if sanitized:
            channel_name = f"{channel_name}-{sanitized}"

        category = await self._get_or_create_project_category(guild)

        try:
            channel = await guild.create_text_channel(
                channel_name,
                category=category,
                topic=f"Meeting room for: {goal[:100]}",
                reason=f"Hyperion project channel for graph {graph_id}",
            )
            logger.info(
                f"Created project channel #{channel_name} (ID: {channel.id})"
            )
            return channel
        except discord.Forbidden:
            logger.error(
                f"No permission to create project channel in guild {guild.name}"
            )
            return None
        except discord.HTTPException as e:
            logger.error(f"Failed to create project channel: {e}")
            return None

    async def delete_project_channel(self, channel_id: int) -> bool:
        """Delete a project meeting room channel."""
        channel = self.bot.get_channel(channel_id)
        if not channel:
            logger.warning(f"Project channel {channel_id} not found for deletion")
            return False

        try:
            channel_name = channel.name
            await channel.delete(reason="Hyperion project plan ended")
            logger.info(
                f"Deleted project channel {channel_name} (ID: {channel_id})"
            )
            return True
        except discord.Forbidden:
            logger.error(f"No permission to delete project channel {channel_id}")
            return False
        except discord.HTTPException as e:
            logger.error(f"Failed to delete project channel {channel_id}: {e}")
            return False

    def is_project_channel(self, channel: discord.abc.GuildChannel) -> bool:
        """Check if a channel is a Hyperion project meeting room."""
        if not hasattr(channel, "name"):
            return False
        return channel.name.startswith(self.PROJECT_CHANNEL_PREFIX)

    # --------------------------------------------------
    # DEPARTMENT CHANNELS (Persistent)
    # --------------------------------------------------

    async def promote_to_department(
        self, channel_id: int, project_name: str
    ) -> bool:
        """
        Rename a meeting room channel to a department channel.

        Returns True if successful.
        """
        channel = self.bot.get_channel(channel_id)
        if not channel:
            logger.warning(f"Channel {channel_id} not found for department promotion")
            return False

        sanitized = re.sub(r"[^a-z0-9-]", "", project_name.lower().replace(" ", "-"))[:30]
        new_name = f"{self.DEPARTMENT_CHANNEL_PREFIX}{sanitized}"

        try:
            await channel.edit(
                name=new_name,
                topic=f"Department: {project_name}",
                reason=f"Promoted to department for project: {project_name}",
            )
            logger.info(
                f"Promoted channel {channel_id} to department: {new_name}"
            )
            return True
        except discord.Forbidden:
            logger.error(f"No permission to rename channel {channel_id}")
            return False
        except discord.HTTPException as e:
            logger.error(f"Failed to promote channel {channel_id}: {e}")
            return False

    async def create_department_channel(
        self, guild: discord.Guild, project_name: str
    ) -> Optional[discord.TextChannel]:
        """Create a new department channel directly."""
        sanitized = re.sub(r"[^a-z0-9-]", "", project_name.lower().replace(" ", "-"))[:30]
        channel_name = f"{self.DEPARTMENT_CHANNEL_PREFIX}{sanitized}"

        category = await self._get_or_create_project_category(guild)

        try:
            channel = await guild.create_text_channel(
                channel_name,
                category=category,
                topic=f"Department: {project_name}",
                reason=f"Hyperion department channel for project: {project_name}",
            )
            logger.info(
                f"Created department channel #{channel_name} (ID: {channel.id})"
            )
            return channel
        except discord.Forbidden:
            logger.error(
                f"No permission to create department channel in guild {guild.name}"
            )
            return None
        except discord.HTTPException as e:
            logger.error(f"Failed to create department channel: {e}")
            return None

    def is_department_channel(self, channel: discord.abc.GuildChannel) -> bool:
        """Check if a channel is a Hyperion department channel."""
        if not hasattr(channel, "name"):
            return False
        return channel.name.startswith(self.DEPARTMENT_CHANNEL_PREFIX)
