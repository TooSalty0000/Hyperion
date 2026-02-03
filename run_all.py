#!/usr/bin/env python3
"""
Run All Bots - Central control for the multi-agent system.

Starts all agent bots and provides graceful shutdown with Ctrl+C.

Usage:
    python run_all.py            # Start all bots
    python run_all.py --vega     # Start only Vega
    python run_all.py --altair   # Start only Altair
    python run_all.py --polaris  # Start only Polaris
    python run_all.py --canopus  # Start only Canopus
    python run_all.py --api      # Also start Polaris API server
    python run_all.py --vega --polaris  # Start Vega and Polaris
    python run_all.py --verbose  # Enable verbose timing logs
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import discord
from discord.ext import commands


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


# Bot-specific colors
BOT_COLORS = {
    "vega": Colors.CYAN,
    "altair": Colors.MAGENTA,
    "polaris": Colors.YELLOW,
    "canopus": Colors.GREEN,
    "run_all": Colors.BLUE,
    "discord": Colors.BRIGHT_BLACK,
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.BRIGHT_BLACK,
        logging.INFO: Colors.WHITE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BRIGHT_RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Get level color
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)

        # Get bot/logger color based on name
        name_lower = record.name.lower()
        bot_color = Colors.WHITE
        for bot_name, color in BOT_COLORS.items():
            if bot_name in name_lower:
                bot_color = color
                break

        # Format timestamp
        timestamp = self.formatTime(record, self.datefmt)

        # Build colored output
        parts = [
            f"{Colors.DIM}{timestamp}{Colors.RESET}",
            f"{bot_color}{record.name}{Colors.RESET}",
            f"{level_color}{record.levelname:<8}{Colors.RESET}",
            f"{record.getMessage()}",
        ]

        return " │ ".join(parts)


# Configure logging with colors
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

# Check for verbose flag early (before full arg parsing)
VERBOSE_MODE = "--verbose" in sys.argv or "-v" in sys.argv

logging.basicConfig(
    level=logging.DEBUG if VERBOSE_MODE else logging.INFO,
    handlers=[handler],
)

# Reduce noise from discord.py (even in verbose mode)
logging.getLogger("discord").setLevel(logging.WARNING)
logging.getLogger("discord.http").setLevel(logging.WARNING)

# In verbose mode, show more detail from our modules
if VERBOSE_MODE:
    # Enable DEBUG for timing-related loggers
    logging.getLogger("shared.llm").setLevel(logging.DEBUG)
    logging.getLogger("shared.tools").setLevel(logging.DEBUG)
    logging.getLogger("shared.agent_queue").setLevel(logging.DEBUG)
    logging.getLogger("shared.discord_utils").setLevel(logging.DEBUG)

logger = logging.getLogger("run_all")


class BotManager:
    """Manages multiple Discord bots with graceful shutdown."""

    def __init__(self):
        self.bots: dict[str, commands.Bot] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.shutdown_event = asyncio.Event()

    def add_bot(self, name: str, bot: commands.Bot, token: str):
        """Register a bot to be managed."""
        self.bots[name] = (bot, token)

    async def start_bot(self, name: str, bot: commands.Bot, token: str):
        """Start a single bot."""
        try:
            logger.info(f"Starting {name} bot...")
            await bot.start(token)
        except discord.LoginFailure:
            logger.error(f"{name}: Invalid Discord token")
        except discord.PrivilegedIntentsRequired:
            logger.error(
                f"{name}: Privileged intents required. "
                f"Enable 'Message Content Intent' in Discord Developer Portal."
            )
        except asyncio.CancelledError:
            logger.info(f"{name} bot task cancelled")
        except Exception as e:
            logger.error(f"{name} bot error: {e}")
        finally:
            if not bot.is_closed():
                await bot.close()

    async def run_all(self):
        """Run all registered bots concurrently with staggered starts."""
        if not self.bots:
            logger.error("No bots registered")
            return

        # Start bots with staggered delays to avoid Discord rate limiting
        # Discord can reject connections when too many bots connect at once from the same IP
        bot_items = list(self.bots.items())
        for i, (name, (bot, token)) in enumerate(bot_items):
            task = asyncio.create_task(self.start_bot(name, bot, token))
            self.tasks[name] = task

            # Add delay between bot starts (except after the last one)
            if i < len(bot_items) - 1:
                await asyncio.sleep(2.0)  # 2 second delay between each bot

        logger.info(f"Started {len(self.tasks)} bot(s): {', '.join(self.tasks.keys())}")
        logger.info("Press Ctrl+C to shutdown all bots")

        # Wait for shutdown signal or all bots to finish
        try:
            await self.shutdown_event.wait()
        except asyncio.CancelledError:
            pass

        await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown all bots."""
        logger.info("Shutting down all bots...")

        # Close all bots
        for name, (bot, _) in self.bots.items():
            if not bot.is_closed():
                logger.info(f"Closing {name}...")
                await bot.close()

        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("All bots shut down")


async def main(
    run_vega: bool = True,
    run_altair: bool = True,
    run_polaris: bool = True,
    run_canopus: bool = True,
    run_api: bool = False,
):
    """Main entry point."""
    manager = BotManager()

    # Set up signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        manager.shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Initialize Vega bot
    if run_vega:
        try:
            from vega.config import get_config as get_vega_config

            vega_config = get_vega_config()
            if vega_config.discord_token:
                # Import and create Vega bot
                intents = discord.Intents.default()
                intents.message_content = True
                intents.reactions = True
                intents.guilds = True

                vega_bot = commands.Bot(
                    command_prefix="!",
                    intents=intents,
                )

                @vega_bot.event
                async def on_ready():
                    logger.info(f"Vega connected as {vega_bot.user}")
                    # Populate member cache from all guilds
                    from shared.discord_utils import get_member_cache
                    cache = get_member_cache()
                    cache.populate_from_guilds(vega_bot.guilds)

                @vega_bot.event
                async def on_command_error(ctx, error):
                    # Silently ignore command not found - handled via on_message
                    if isinstance(error, commands.CommandNotFound):
                        return
                    logger.error(f"Vega command error: {error}")

                # Load Vega cog
                async def setup_vega():
                    await vega_bot.load_extension("vega.cogs.core")

                vega_bot.setup_hook = setup_vega
                manager.add_bot("Vega", vega_bot, vega_config.discord_token)
            else:
                logger.warning("DISCORD_TOKEN not set, skipping Vega")
        except Exception as e:
            logger.error(f"Failed to initialize Vega: {e}")

    # Initialize Altair bot
    if run_altair:
        try:
            from altair.config import get_config as get_altair_config
            from altair.bot import AltairBot

            altair_config = get_altair_config()
            if altair_config.discord_token:
                altair_bot = AltairBot()
                manager.add_bot("Altair", altair_bot, altair_config.discord_token)
            else:
                logger.warning("ALTAIR_DISCORD_TOKEN not set, skipping Altair")
        except Exception as e:
            logger.error(f"Failed to initialize Altair: {e}")

    # Initialize Polaris bot
    if run_polaris:
        try:
            from polaris.config import get_config as get_polaris_config
            from polaris.bot import PolarisBot

            polaris_config = get_polaris_config()
            if polaris_config.discord_token:
                polaris_bot = PolarisBot()
                manager.add_bot("Polaris", polaris_bot, polaris_config.discord_token)
            else:
                logger.warning("POLARIS_DISCORD_TOKEN not set, skipping Polaris")
        except Exception as e:
            logger.error(f"Failed to initialize Polaris: {e}")

    # Initialize Canopus bot
    if run_canopus:
        try:
            from canopus.config import get_config as get_canopus_config
            from canopus.bot import CanopusBot

            canopus_config = get_canopus_config()
            if canopus_config.discord_token:
                canopus_bot = CanopusBot()
                manager.add_bot("Canopus", canopus_bot, canopus_config.discord_token)
            else:
                logger.warning("CANOPUS_DISCORD_TOKEN not set, skipping Canopus")
        except Exception as e:
            logger.error(f"Failed to initialize Canopus: {e}")

    if not manager.bots and not run_api:
        logger.error("No bots could be initialized. Check your .env configuration.")
        return

    # Optionally start Polaris API server alongside bots
    api_task = None
    if run_api:
        try:
            import uvicorn
            from polaris.api.config import get_api_config

            api_config = get_api_config()
            uv_config = uvicorn.Config(
                "polaris.api.main:app",
                host=api_config.host,
                port=api_config.port,
                log_level="info",
            )
            server = uvicorn.Server(uv_config)
            api_task = asyncio.create_task(server.serve())
            logger.info(f"Polaris API starting on {api_config.host}:{api_config.port}")
        except ImportError:
            logger.error("uvicorn not installed — cannot start API server")
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")

    await manager.run_all()

    # Clean up API task
    if api_task and not api_task.done():
        api_task.cancel()
        try:
            await api_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    # Parse simple args - if any specific bot is mentioned, only run those
    has_specific_bot = any(
        arg in sys.argv for arg in ["--vega", "--altair", "--polaris", "--canopus"]
    )

    run_api = "--api" in sys.argv

    if has_specific_bot:
        run_vega = "--vega" in sys.argv
        run_altair = "--altair" in sys.argv
        run_polaris = "--polaris" in sys.argv
        run_canopus = "--canopus" in sys.argv
    else:
        # No specific bot mentioned - run all
        run_vega = True
        run_altair = True
        run_polaris = True
        run_canopus = True

    if VERBOSE_MODE:
        logger.info("🔍 Verbose mode enabled - showing timing details")

    try:
        asyncio.run(main(
            run_vega=run_vega,
            run_altair=run_altair,
            run_polaris=run_polaris,
            run_canopus=run_canopus,
            run_api=run_api,
        ))
    except KeyboardInterrupt:
        logger.info("Interrupted")
