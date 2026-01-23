"""Discord message preprocessing utilities.

Converts raw Discord messages into clean, LLM-readable text by resolving
mentions, embeds, attachments, and other Discord-specific formatting.
"""

import re
import logging
from typing import Optional

import discord

logger = logging.getLogger(__name__)


async def resolve_message_content(
    msg: discord.Message,
    bot: discord.Client,
    agent_registry: dict[str, int],
) -> str:
    """
    Convert a Discord message into clean, LLM-readable text.

    Resolves:
    - <@123456> and <@!123456> → @AgentName or @Username
    - <#123456> → #channel-name
    - <@&123456> → @RoleName
    - Embeds → [Embed: title | description]
    - Attachments → [Attachment: filename.ext]
    - Reply references → [replying to @Name]
    - Strips internal protocol markers like `[from: agent_name]`

    Args:
        msg: Discord message to process
        bot: Discord bot/client for resolving users/channels
        agent_registry: Mapping of agent names to Discord user IDs

    Returns:
        Clean text suitable for LLM consumption
    """
    content = msg.content

    # Build reverse registry: user_id → agent_name
    id_to_agent = {uid: name for name, uid in agent_registry.items()}

    # Resolve user mentions: <@123456> or <@!123456>
    async def resolve_user_mention(match: re.Match) -> str:
        user_id = int(match.group(1))
        # Check if it's a known agent
        if user_id in id_to_agent:
            return f"@{id_to_agent[user_id].title()}"
        # Try to resolve from Discord
        try:
            user = bot.get_user(user_id)
            if user:
                return f"@{user.display_name}"
            user = await bot.fetch_user(user_id)
            return f"@{user.display_name}"
        except Exception:
            return f"@User({user_id})"

    # Process user mentions
    mention_pattern = re.compile(r'<@!?(\d+)>')
    matches = list(mention_pattern.finditer(content))
    # Process in reverse order to preserve positions
    for match in reversed(matches):
        resolved = await resolve_user_mention(match)
        content = content[:match.start()] + resolved + content[match.end():]

    # Resolve channel mentions: <#123456>
    def resolve_channel_mention(match: re.Match) -> str:
        channel_id = int(match.group(1))
        channel = bot.get_channel(channel_id)
        if channel and hasattr(channel, 'name'):
            return f"#{channel.name}"
        return f"#channel({channel_id})"

    content = re.sub(r'<#(\d+)>', resolve_channel_mention, content)

    # Resolve role mentions: <@&123456>
    def resolve_role_mention(match: re.Match) -> str:
        role_id = int(match.group(1))
        if msg.guild:
            role = msg.guild.get_role(role_id)
            if role:
                return f"@{role.name}"
        return f"@role({role_id})"

    content = re.sub(r'<@&(\d+)>', resolve_role_mention, content)

    # Strip internal protocol markers: `[from: agent_name]`
    content = re.sub(r'\n?`\[from: \w+\]`', '', content)

    # Strip request type/project markers from MentionAgentTool
    content = re.sub(r'\[(URGENT|ACTION|STATUS|QUESTION|TASK)\]\s*', '', content)
    content = re.sub(r'\[Project: [^\]]+\]\s*', '', content)

    # Handle embeds
    embed_parts = []
    for embed in msg.embeds:
        parts = []
        if embed.title:
            parts.append(embed.title)
        if embed.description:
            parts.append(embed.description)
        if parts:
            embed_parts.append(f"[Embed: {' | '.join(parts)}]")
        # Include fields
        for field in embed.fields:
            embed_parts.append(f"[{field.name}: {field.value}]")

    # Handle attachments
    attachment_parts = []
    for att in msg.attachments:
        attachment_parts.append(f"[Attachment: {att.filename}]")

    # Handle reply reference
    reply_prefix = ""
    if msg.reference and msg.reference.resolved:
        ref_msg = msg.reference.resolved
        if isinstance(ref_msg, discord.Message):
            ref_author = ref_msg.author.display_name
            reply_prefix = f"[replying to @{ref_author}] "

    # Combine all parts
    result = reply_prefix + content.strip()
    if embed_parts:
        result += "\n" + "\n".join(embed_parts)
    if attachment_parts:
        result += "\n" + "\n".join(attachment_parts)

    return result.strip()


def get_author_display_name(
    msg: discord.Message,
    agent_registry: dict[str, int],
) -> str:
    """
    Get a clean display name for a message author.

    For agents, returns the agent name (capitalized).
    For users, returns the Discord display name.

    Args:
        msg: Discord message
        agent_registry: Mapping of agent names to Discord user IDs

    Returns:
        Display name string
    """
    # Check if author is a known agent
    for name, uid in agent_registry.items():
        if msg.author.id == uid:
            return name.title()

    return msg.author.display_name


def is_from_known_agent(
    msg: discord.Message,
    agent_registry: dict[str, int],
) -> Optional[str]:
    """
    Check if a message is from a known agent.

    Args:
        msg: Discord message
        agent_registry: Mapping of agent names to Discord user IDs

    Returns:
        Agent name if from a known agent, None otherwise
    """
    for name, uid in agent_registry.items():
        if msg.author.id == uid:
            return name
    return None
