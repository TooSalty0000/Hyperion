"""Polaris Discord Bot - Entry point for the Polaris agent."""

import asyncio
import logging
import sys
from pathlib import Path

import discord
from discord.ext import commands

from polaris.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("polaris")


class PolarisBot(commands.Bot):
    """Polaris Discord Bot."""

    def __init__(self):
        self.config = get_config()

        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.guilds = True

        super().__init__(
            command_prefix=self.config.command_prefix,
            intents=intents,
            help_command=commands.DefaultHelpCommand(),
        )

    async def setup_hook(self):
        """Called when bot is ready to set up."""
        logger.info("Loading Polaris cogs...")

        # Load the core cog
        await self.load_extension("polaris.cogs.core")

        logger.info("Polaris cogs loaded successfully")

    async def on_ready(self):
        """Called when bot is fully connected and ready."""
        logger.info(f"Polaris connected as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")

        # Populate member cache from all guilds
        from shared.discord_utils import get_member_cache
        cache = get_member_cache()
        cache.populate_from_guilds(self.guilds)

        # Check calendar authentication status
        await self._check_calendar_status()

        # Set status based on calendar connection
        status_text = "for @mentions | Calendar specialist"
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=status_text,
            )
        )

    async def _check_calendar_status(self):
        """Check and log calendar authentication status on startup."""
        from polaris.tools.utils import check_calendar_auth_status

        auth_status = check_calendar_auth_status(
            credentials_path=self.config.google_credentials_path,
            token_path=self.config.google_token_path,
        )

        if auth_status['authenticated']:
            logger.info("✅ Calendar: OAuth token found and valid")
            logger.info(f"   Token path: {auth_status['token_path']}")
        else:
            logger.warning("⚠️  Calendar: Not authenticated")
            logger.warning(f"   {auth_status['message']}")
            logger.warning(f"   Token path: {auth_status['token_path']}")
            if not auth_status['credentials_exists']:
                logger.warning("   Set GOOGLE_CREDENTIALS_PATH to enable calendar features")

    async def on_connect(self):
        """Called when bot connects to Discord."""
        logger.info("Polaris connected to Discord gateway")

    async def on_disconnect(self):
        """Called when bot disconnects from Discord."""
        logger.warning("Polaris disconnected from Discord gateway")

    async def on_error(self, event_method: str, *args, **kwargs):
        """Handle errors in event handlers."""
        logger.exception(f"Error in {event_method}")

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError):
        """Handle command errors silently for commands that don't exist."""
        if isinstance(error, commands.CommandNotFound):
            return

        logger.error(f"Command error in {ctx.command}: {error}")


def run_bot():
    """Run the Polaris bot."""
    config = get_config()

    if not config.discord_token:
        logger.error("POLARIS_DISCORD_TOKEN not set in environment")
        sys.exit(1)

    bot = PolarisBot()

    try:
        logger.info("Starting Polaris bot...")
        bot.run(config.discord_token, log_handler=None)
    except discord.LoginFailure:
        logger.error("Invalid Discord token for Polaris")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running Polaris bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_bot()
