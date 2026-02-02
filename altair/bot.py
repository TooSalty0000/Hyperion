"""Altair Discord Bot - Entry point for the Altair agent."""

import asyncio
import logging
import sys
from pathlib import Path

import discord
from discord.ext import commands

from altair.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("altair")


class AltairBot(commands.Bot):
    """Altair Discord Bot."""

    def __init__(self):
        self.config = get_config()

        # Set up intents
        # Note: Message Content Intent must be enabled in Discord Developer Portal
        intents = discord.Intents.default()
        intents.message_content = True  # REQUIRED - enable in Developer Portal
        intents.reactions = True
        intents.guilds = True
        # intents.members = True  # Only enable if needed, requires Developer Portal

        super().__init__(
            command_prefix=self.config.command_prefix,
            intents=intents,
            help_command=commands.DefaultHelpCommand(),
        )

    async def setup_hook(self):
        """Called when bot is ready to set up."""
        logger.info("Loading Altair cogs...")

        # Load the core cog
        await self.load_extension("altair.cogs.core")

        logger.info("Altair cogs loaded successfully")

    async def on_ready(self):
        """Called when bot is fully connected and ready."""
        logger.info(f"Altair connected as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")

        # Populate member cache from all guilds
        from shared.discord_utils import get_member_cache
        cache = get_member_cache()
        cache.populate_from_guilds(self.guilds)

        # Set status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for @mentions | CLI specialist",
            )
        )

    async def on_connect(self):
        """Called when bot connects to Discord."""
        logger.info("Altair connected to Discord gateway")

    async def on_disconnect(self):
        """Called when bot disconnects from Discord."""
        logger.warning("Altair disconnected from Discord gateway")

    async def on_error(self, event_method: str, *args, **kwargs):
        """Handle errors in event handlers."""
        logger.exception(f"Error in {event_method}")

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError):
        """Handle command errors silently for commands that don't exist.

        This prevents noisy logs when messages with ! prefix are meant for
        other bots (like Vega's !vli commands).
        """
        if isinstance(error, commands.CommandNotFound):
            # Silently ignore - this is likely a command for another bot
            return

        # Log other command errors
        logger.error(f"Command error in {ctx.command}: {error}")


def run_bot():
    """Run the Altair bot."""
    config = get_config()

    if not config.discord_token:
        logger.error("ALTAIR_DISCORD_TOKEN not set in environment")
        sys.exit(1)

    bot = AltairBot()

    try:
        logger.info("Starting Altair bot...")
        bot.run(config.discord_token, log_handler=None)
    except discord.LoginFailure:
        logger.error("Invalid Discord token for Altair")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running Altair bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_bot()
