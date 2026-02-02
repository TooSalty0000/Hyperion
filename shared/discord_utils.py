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


class MemberCache:
    """
    Cache of Discord guild members for fast @mention lookups.

    Initialized at bot startup and can be refreshed dynamically.
    """

    def __init__(self):
        # name (lowercase) -> user_id
        self._by_name: dict[str, int] = {}
        # user_id -> display_name (for reverse lookups)
        self._by_id: dict[int, str] = {}
        self._initialized = False

    def populate_from_guilds(self, guilds):
        """Populate cache from all bot guilds. Call this in on_ready."""
        count = 0
        for guild in guilds:
            for member in guild.members:
                self._add_member(member)
                count += 1
        self._initialized = True
        logger.info(f"[MemberCache] Populated with {count} members from {len(list(guilds))} guilds")

    def _add_member(self, member: discord.Member):
        """Add a single member to the cache."""
        # Index by display name and username (both lowercase)
        self._by_name[member.display_name.lower()] = member.id
        self._by_name[member.name.lower()] = member.id
        self._by_id[member.id] = member.display_name

    def add_member(self, member: discord.Member):
        """Public method to add a member (e.g., when a new member joins)."""
        self._add_member(member)
        logger.debug(f"[MemberCache] Added member: {member.display_name} ({member.id})")

    def get_user_id(self, name: str) -> Optional[int]:
        """Look up user ID by name (case-insensitive)."""
        return self._by_name.get(name.lower())

    def get_display_name(self, user_id: int) -> Optional[str]:
        """Look up display name by user ID."""
        return self._by_id.get(user_id)

    @property
    def member_count(self) -> int:
        """Number of unique members in cache."""
        return len(self._by_id)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# Global member cache - shared across all cogs
_member_cache: Optional[MemberCache] = None


def get_member_cache() -> MemberCache:
    """Get the global member cache (creates one if needed)."""
    global _member_cache
    if _member_cache is None:
        _member_cache = MemberCache()
    return _member_cache


async def convert_text_mentions_to_discord(
    content: str,
    channel: discord.abc.Messageable,
    agent_registry: dict[str, int],
) -> str:
    """
    Convert @Name text patterns to actual Discord mentions.

    Searches for patterns like `@Sirius`, `@Vega`, or `@Username` and converts
    them to Discord mention format `<@user_id>` if the name matches:
    1. The global member cache (populated at startup from all guilds)
    2. A known agent in the registry (fallback)
    3. Live guild member lookup (if not in cache)

    Args:
        content: Message content to process
        channel: Discord channel (used for live lookups if cache miss)
        agent_registry: Mapping of agent names to Discord user IDs (fallback)

    Returns:
        Content with @Name patterns replaced with Discord mentions
    """
    if not content or "@" not in content:
        return content

    cache = get_member_cache()

    # Build case-insensitive agent lookup as fallback
    agent_lookup = {name.lower(): uid for name, uid in agent_registry.items()}

    logger.info(
        f"[MentionConvert] Processing: cache={cache.member_count} members, "
        f"agents={list(agent_lookup.keys())}, content='{content[:60]}...'"
    )

    # Get guild for live lookups if cache misses
    guild = getattr(channel, 'guild', None) if hasattr(channel, 'guild') else None

    # Pattern to find @Name (word characters after @, not already a Discord mention)
    pattern = re.compile(r'(?<![<\w])@(\w+)(?!>)')

    # We need async lookup, so collect matches first
    matches = list(pattern.finditer(content))
    if not matches:
        return content

    # Process matches in reverse order to preserve string positions
    result = content
    for match in reversed(matches):
        name = match.group(1)
        name_lower = name.lower()
        user_id = None
        source = None

        # 1. Check member cache first (fastest)
        user_id = cache.get_user_id(name_lower)
        if user_id:
            source = "cache"

        # 2. Check agent registry (fallback for bots)
        if not user_id and name_lower in agent_lookup:
            user_id = agent_lookup[name_lower]
            source = "agent"

        # 3. Try live guild member search (updates cache)
        if not user_id and guild:
            # Search by name
            member = guild.get_member_named(name)
            if not member:
                # Try case-insensitive search
                for m in guild.members:
                    if m.display_name.lower() == name_lower or m.name.lower() == name_lower:
                        member = m
                        break

            if member:
                user_id = member.id
                source = "guild"
                # Update cache for next time
                cache.add_member(member)

        # 4. Try fetching by query (last resort - API call)
        if not user_id and guild:
            try:
                members = await guild.query_members(query=name, limit=5)
                for m in members:
                    if m.display_name.lower() == name_lower or m.name.lower() == name_lower:
                        user_id = m.id
                        source = "query"
                        cache.add_member(m)
                        break
            except Exception as e:
                logger.debug(f"[MentionConvert] query_members failed: {e}")

        # Replace if found
        if user_id:
            logger.info(f"[MentionConvert] ✓ @{name} -> <@{user_id}> ({source})")
            result = result[:match.start()] + f"<@{user_id}>" + result[match.end():]
        else:
            logger.warning(f"[MentionConvert] ✗ @{name} not found")

    return result
