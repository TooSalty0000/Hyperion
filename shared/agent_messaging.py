"""Inter-agent communication via Discord @mentions."""

import logging
from typing import Optional, Dict, Any

import discord

logger = logging.getLogger(__name__)


class AgentMessaging:
    """
    Helper for inter-agent Discord communication.

    Agents communicate by posting messages that @mention each other.
    This is visible to users in Discord, unlike the old AgentBus.
    """

    def __init__(
        self,
        bot: discord.Client,
        agent_registry: Dict[str, int],  # {"vega": user_id, "altair": user_id}
        own_agent_name: str,
    ):
        """
        Initialize agent messaging.

        Args:
            bot: Discord bot client
            agent_registry: Mapping of agent names to Discord user IDs
            own_agent_name: Name of the agent using this messaging instance
        """
        self.bot = bot
        self.agent_registry = {k.lower(): v for k, v in agent_registry.items()}
        self.own_agent_name = own_agent_name.lower()

    async def mention_agent(
        self,
        channel: discord.TextChannel,
        target_agent: str,
        message: str,
        **kwargs
    ) -> Optional[discord.Message]:
        """
        Send a message that @mentions another agent.

        The message will be visible in the Discord channel, and the
        target agent's bot will receive it via on_message.

        Args:
            channel: Discord channel to post in
            target_agent: Name of the agent to mention (e.g., "altair")
            message: The message content

        Returns:
            The sent Discord message, or None if failed
        """
        target_name = target_agent.lower()
        target_id = self.agent_registry.get(target_name)

        if not target_id:
            logger.error(f"Unknown agent: {target_agent}")
            return None

        # Format message with mention
        mention = f"<@{target_id}>"
        full_message = f"{mention} {message}"

        try:
            sent_message = await channel.send(full_message, **kwargs)
            logger.info(f"{self.own_agent_name} mentioned {target_agent}: {message[:50]}...")
            return sent_message
        except Exception as e:
            logger.error(f"Failed to send mention to {target_agent}: {e}")
            return None

    async def reply_to_agent(
        self,
        original_message: discord.Message,
        content: str,
        mention_sender: bool = False,
        **kwargs
    ) -> Optional[discord.Message]:
        """
        Reply to a message from another agent.

        Args:
            original_message: The message to reply to
            content: Reply content
            mention_sender: Whether to @mention the sender in the reply

        Returns:
            The sent Discord message, or None if failed
        """
        try:
            if mention_sender:
                sender_id = original_message.author.id
                content = f"<@{sender_id}> {content}"

            sent_message = await original_message.channel.send(
                content,
                reference=original_message,
                **kwargs
            )
            return sent_message
        except Exception as e:
            logger.error(f"Failed to reply: {e}")
            return None

    def is_from_agent(self, message: discord.Message) -> Optional[str]:
        """
        Check if a message is from a known agent.

        Args:
            message: Discord message to check

        Returns:
            Agent name if from an agent, None otherwise
        """
        for name, user_id in self.agent_registry.items():
            if message.author.id == user_id:
                return name
        return None

    def get_mentioned_agents(self, message: discord.Message) -> list[str]:
        """
        Get list of agents mentioned in a message.

        Args:
            message: Discord message to check

        Returns:
            List of agent names that were mentioned
        """
        mentioned = []
        for name, user_id in self.agent_registry.items():
            for mention in message.mentions:
                if mention.id == user_id:
                    mentioned.append(name)
                    break
        return mentioned

    def extract_message_for_agent(
        self,
        message: discord.Message,
        agent_name: str
    ) -> str:
        """
        Extract the message content intended for a specific agent.

        Removes the @mention from the message content.

        Args:
            message: Discord message
            agent_name: Name of the agent to extract content for

        Returns:
            Message content with the agent's mention removed
        """
        agent_id = self.agent_registry.get(agent_name.lower())
        if not agent_id:
            return message.content

        content = message.content
        # Remove mention formats
        content = content.replace(f"<@{agent_id}>", "").strip()
        content = content.replace(f"<@!{agent_id}>", "").strip()

        return content
