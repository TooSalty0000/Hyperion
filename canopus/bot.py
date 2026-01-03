"""Canopus Discord Bot - Entry point for the Canopus agent."""

import asyncio
import logging
import sys
from pathlib import Path

import discord
from discord.ext import commands

from canopus.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("canopus")


class CanopusBot(commands.Bot):
    """Canopus Discord Bot - Web intelligence and browser automation."""

    def __init__(self):
        self.config = get_config()

        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True  # REQUIRED - enable in Developer Portal
        intents.reactions = True
        intents.guilds = True

        super().__init__(
            command_prefix=self.config.command_prefix,
            intents=intents,
            help_command=commands.DefaultHelpCommand(),
        )

    async def setup_hook(self):
        """Called when bot is ready to set up."""
        logger.info("Loading Canopus cogs...")

        # Load the core cog
        await self.load_extension("canopus.cogs.core")

        logger.info("Canopus cogs loaded successfully")

    async def on_ready(self):
        """Called when bot is fully connected and ready."""
        logger.info(f"Canopus connected as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")

        # Set status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for @mentions | Web specialist",
            )
        )

    async def on_connect(self):
        """Called when bot connects to Discord."""
        logger.info("Canopus connected to Discord gateway")

    async def on_disconnect(self):
        """Called when bot disconnects from Discord."""
        logger.warning("Canopus disconnected from Discord gateway")

    async def on_error(self, event_method: str, *args, **kwargs):
        """Handle errors in event handlers."""
        logger.exception(f"Error in {event_method}")

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError):
        """Handle command errors silently for commands that don't exist."""
        if isinstance(error, commands.CommandNotFound):
            # Silently ignore - likely a command for another bot
            return

        logger.error(f"Command error in {ctx.command}: {error}")


def run_bot():
    """Run the Canopus bot."""
    config = get_config()

    if not config.discord_token:
        logger.error("CANOPUS_DISCORD_TOKEN not set in environment")
        sys.exit(1)

    bot = CanopusBot()

    try:
        logger.info("Starting Canopus bot...")
        bot.run(config.discord_token, log_handler=None)
    except discord.LoginFailure:
        logger.error("Invalid Discord token for Canopus")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running Canopus bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_bot()
