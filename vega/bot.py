"""Vega Discord Bot - Entry point for the Vega orchestrator agent."""

import logging
import sys

import discord
from discord.ext import commands

from vega.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("vega")


class VegaBot(commands.Bot):
    """Vega Discord Bot - Orchestrator agent."""

    def __init__(self):
        self.config = get_config()

        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.guilds = True

        super().__init__(
            command_prefix="!",
            intents=intents,
        )

    async def setup_hook(self):
        """Called when bot is ready to set up."""
        logger.info("Loading Vega cogs...")
        await self.load_extension("vega.cogs.core")
        logger.info("Vega cogs loaded successfully")

    async def on_ready(self):
        """Called when bot is fully connected and ready."""
        logger.info(f"Vega connected as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guild(s)")

        # Populate member cache from all guilds
        from shared.discord_utils import get_member_cache
        cache = get_member_cache()
        cache.populate_from_guilds(self.guilds)

    async def on_connect(self):
        """Called when bot connects to Discord."""
        logger.info("Vega connected to Discord gateway")

    async def on_disconnect(self):
        """Called when bot disconnects from Discord."""
        logger.warning("Vega disconnected from Discord gateway")

    async def on_error(self, event_method: str, *args, **kwargs):
        """Handle errors in event handlers."""
        logger.exception(f"Error in {event_method}")

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError):
        """Handle command errors silently for commands that don't exist."""
        if isinstance(error, commands.CommandNotFound):
            return
        logger.error(f"Command error in {ctx.command}: {error}")


def run_bot():
    """Run the Vega bot."""
    config = get_config()

    if not config.discord_token:
        logger.error("DISCORD_TOKEN not set in environment")
        sys.exit(1)

    bot = VegaBot()

    try:
        logger.info("Starting Vega bot...")
        bot.run(config.discord_token, log_handler=None)
    except discord.LoginFailure:
        logger.error("Invalid Discord token for Vega")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running Vega bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_bot()
