"""Unified notification manager — routes to Discord DM + Web Push."""

import logging
from typing import Optional

import discord

from polaris.notifications.push import PushNotificationService
from polaris.todo.reminder_models import ReminderType, ReminderItem

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Routes notifications to appropriate channels:
    - Discord DMs (via discord.py bot)
    - Web Push (via pywebpush / VAPID)
    - Channel mentions (via Discord channel)

    Centralizes notification delivery so reminders, adaptation engine,
    and other subsystems don't need to handle transport details.
    """

    def __init__(
        self,
        bot: discord.Client,
        push_service: Optional[PushNotificationService] = None,
        fallback_user_id: Optional[int] = None,
        default_channel_id: Optional[int] = None,
    ):
        self._bot = bot
        self._push_service = push_service
        self._fallback_user_id = fallback_user_id
        self._default_channel_id = default_channel_id

    @property
    def push_available(self) -> bool:
        return self._push_service is not None and self._push_service.is_configured

    def _resolve_channel_id(self, reminder_channel_id: Optional[int] = None) -> Optional[int]:
        """Get the channel to post reminders in. Tries: reminder-specific → configured default → first guild text channel."""
        if reminder_channel_id:
            return reminder_channel_id
        if self._default_channel_id:
            return self._default_channel_id
        # Auto-discover from first guild
        for guild in self._bot.guilds:
            if guild.system_channel:
                return guild.system_channel.id
            for ch in guild.text_channels:
                return ch.id
        return None

    async def send_reminder(self, reminder: ReminderItem, extra_text: str = "", formatted_text: str = ""):
        """Send a reminder notification: @mention in channel + Web Push.

        Args:
            reminder: The reminder item to send.
            extra_text: Additional context (e.g. linked todo info).
            formatted_text: LLM-generated natural text. Falls back to
                            "**Reminder:** {message}" if empty.
        """
        results = {}
        user_id = str(self._fallback_user_id) if self._fallback_user_id else reminder.user_id

        # Build channel text: prefer LLM-generated text, fall back to hard-coded format
        if formatted_text:
            channel_text = formatted_text
            if extra_text:
                channel_text = f"{channel_text}\n{extra_text}"
        else:
            body = reminder.message
            if extra_text:
                body = f"{body}\n{extra_text}"
            channel_text = f"**Reminder:** {body}"

        # Channel @mention (primary delivery)
        channel_id = self._resolve_channel_id(reminder.channel_id)
        if channel_id:
            results["channel"] = await self._send_channel_mention(
                channel_id=channel_id,
                user_id=user_id,
                text=channel_text,
            )
        else:
            logger.warning("No channel found for reminder notification")

        # Web Push
        if self.push_available:
            push_body = formatted_text if formatted_text else reminder.message
            results["push"] = await self._push_service.send_notification(
                title="Reminder",
                body=push_body,
                tag=f"reminder-{reminder.id}",
                data={"reminder_id": reminder.id, "todo_id": reminder.todo_id},
            )

        return results

    async def send_alert(
        self,
        user_id: str,
        title: str,
        body: str,
        channel_id: Optional[int] = None,
        use_push: bool = True,
    ):
        """Send a general alert notification (e.g., from adaptation engine)."""
        results = {}

        # Always try DM
        results["dm"] = await self._send_discord_dm(user_id=user_id, text=f"**{title}**\n{body}")

        # Channel if specified
        if channel_id:
            results["channel"] = await self._send_channel_mention(
                channel_id=channel_id,
                user_id=user_id,
                text=f"**{title}**\n{body}",
            )

        # Web Push
        if use_push and self.push_available:
            results["push"] = await self._push_service.send_notification(
                title=title,
                body=body,
                tag="alert",
            )

        return results

    async def _send_discord_dm(self, user_id: str, text: str) -> bool:
        """Send a Discord DM to the app owner. Always uses fallback_user_id (single-user system)."""
        try:
            # Always DM the configured owner — stored user_id may be a bot ID or "owner"
            uid = self._fallback_user_id
            if not uid:
                uid = int(user_id) if isinstance(user_id, str) and user_id.isdigit() else None
            if not uid:
                logger.warning(f"No valid user_id for DM: {user_id} (no fallback configured)")
                return False

            user = await self._bot.fetch_user(uid)
            if user:
                await user.send(text)
                return True
            return False
        except discord.Forbidden:
            logger.warning(f"Cannot DM user {uid} — DMs disabled or blocked")
            return False
        except Exception as e:
            logger.error(f"DM send failed for {uid}: {e}")
            return False

    async def _send_channel_mention(self, channel_id: int, user_id: str, text: str) -> bool:
        """Send a mention message in a Discord channel."""
        try:
            channel = self._bot.get_channel(channel_id)
            if not channel:
                return False
            await channel.send(f"<@{user_id}> {text}")
            return True
        except Exception as e:
            logger.error(f"Channel send failed for {channel_id}: {e}")
            return False
