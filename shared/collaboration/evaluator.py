"""Chime-in evaluation for collaborative agent responses."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# Mapping from text descriptions to actual emoji
REACTION_EMOJI_MAP = {
    "checkmark": "\u2705",
    "check": "\u2705",
    "thumbsup": "\U0001F44D",
    "thumbs_up": "\U0001F44D",
    "thumbs up": "\U0001F44D",
    "ok": "\U0001F44C",
    "wave": "\U0001F44B",
    "eyes": "\U0001F440",
    "rocket": "\U0001F680",
    "star": "\u2B50",
    "heart": "\u2764\uFE0F",
    "fire": "\U0001F525",
    "100": "\U0001F4AF",
    "clap": "\U0001F44F",
    "pray": "\U0001F64F",
    "salute": "\U0001FAE1",
}


def _parse_reaction(reaction_text: str) -> str:
    """Convert reaction text to actual emoji."""
    if not reaction_text:
        return "\u2705"  # Default: checkmark

    text = reaction_text.lower().strip()

    # If it's already an emoji, return it
    if len(text) <= 2 or text.startswith(":"):
        # Check if it's a Discord-style :emoji: format
        if text.startswith(":") and text.endswith(":"):
            text = text[1:-1]

    return REACTION_EMOJI_MAP.get(text, "\u2705")


# Structured output schema for chime-in decisions (Gemini)
CHIME_IN_DECISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "decision": {
            "type": "STRING",
            "enum": ["RESPOND", "ACKNOWLEDGE_ONLY", "STAY_QUIET", "NOT_RELEVANT"],
            "description": "Whether to chime in on this message"
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence level from 0.0 to 1.0"
        },
        "reasoning": {
            "type": "STRING",
            "description": "Brief explanation for the decision"
        },
        "reaction": {
            "type": "STRING",
            "description": "Emoji reaction for ACKNOWLEDGE_ONLY (e.g., 'checkmark', 'thumbsup')"
        }
    },
    "required": ["decision", "confidence", "reasoning"]
}


class ChimeInDecision(Enum):
    """Decision about whether to chime in on a conversation."""
    RESPOND = "respond"        # Agent should respond with full message
    ACKNOWLEDGE_ONLY = "acknowledge"  # Just react with emoji, no text needed
    STAY_QUIET = "stay_quiet"  # Agent decided not to respond
    NOT_RELEVANT = "not_relevant"  # Not relevant to this agent's domain


@dataclass
class ChimeInContext:
    """
    Context for evaluating whether an agent should chime in.

    Provided to agents when a message appears in the main channel
    that they weren't directly mentioned in.
    """
    # Message details
    channel_id: int
    message_content: str
    message_author: str
    message_author_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    # Agent context
    is_from_agent: bool = False
    mentioned_agents: List[str] = field(default_factory=list)

    # Conversation context (recent messages for context)
    recent_messages: List[str] = field(default_factory=list)

    # Event context (if this is from an event)
    event_source: Optional[str] = None
    event_type: Optional[str] = None


@dataclass
class ChimeInResult:
    """
    Result of a chime-in evaluation.

    Contains the agent's decision and optional response.
    """
    decision: ChimeInDecision
    response: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: float = 0.5  # 0.0 to 1.0
    suggested_reaction: Optional[str] = None  # Emoji for ACKNOWLEDGE_ONLY

    @property
    def should_respond(self) -> bool:
        """Whether the agent should respond based on this result."""
        return self.decision == ChimeInDecision.RESPOND

    @property
    def should_acknowledge(self) -> bool:
        """Whether the agent should just add a reaction."""
        return self.decision == ChimeInDecision.ACKNOWLEDGE_ONLY


# Evaluation prompt template for chime-in decisions (utility LLM - decision only)
CHIME_IN_EVALUATION_PROMPT = """INTERNAL DECISION PROCESS

You are the decision module for {agent_name}.

=== INPUT ===
Message: "{message}"
From: {author}
Your domain: {agent_description}
Mentioned agents: {mentioned_agents}
Recent conversation:
{recent_context}
Memories: {memory_context}

=== CRITICAL MENTION RULES (check FIRST) ===
- If mentioned_agents is NOT empty AND you are NOT in the list → STAY_QUIET (message is for someone else)
- If mentioned_agents IS empty → Evaluate based on content relevance
- If you ARE in mentioned_agents → Evaluate whether RESPOND or ACKNOWLEDGE_ONLY is appropriate

=== DECISION OPTIONS ===
- RESPOND: You need to take action or provide substantial information
- ACKNOWLEDGE_ONLY: Just react with emoji (for FYI messages, handoff completions, status updates where no action needed)
- STAY_QUIET: Not relevant to you OR already handled OR message is for someone else

=== RESPOND triggers (only if you pass mention rules) ===
- Words like "everyone", "all of you" addressing the group
- Your name mentioned and action/response is expected
- Direct question requiring your expertise

=== ACKNOWLEDGE_ONLY triggers ===
- You are mentioned but it's just an FYI or status update
- Handoff completion ("@Agent, done with X")
- No action or response is actually needed from you

=== OUTPUT FORMAT ===
DECISION: RESPOND | ACKNOWLEDGE_ONLY | STAY_QUIET
CONFIDENCE: 0.9
REASONING: [brief reason]
REACTION: [emoji, only for ACKNOWLEDGE_ONLY, e.g. "checkmark" or "thumbsup"]
"""




def parse_chime_in_response(response_text: str) -> ChimeInResult:
    """
    Parse the LLM's chime-in evaluation response.

    Handles:
    1. JSON format (from structured output) - preferred
    2. Text format (DECISION: RESPOND) - fallback
    3. Natural language - last resort

    Args:
        response_text: Raw text response from LLM

    Returns:
        ChimeInResult with parsed decision
    """
    # Try JSON format first (from structured output)
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, dict) and "decision" in data:
            decision_str = data["decision"].upper()
            if decision_str == "RESPOND":
                decision = ChimeInDecision.RESPOND
            elif decision_str == "ACKNOWLEDGE_ONLY":
                decision = ChimeInDecision.ACKNOWLEDGE_ONLY
            elif decision_str == "NOT_RELEVANT":
                decision = ChimeInDecision.NOT_RELEVANT
            else:
                decision = ChimeInDecision.STAY_QUIET

            return ChimeInResult(
                decision=decision,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                suggested_reaction=_parse_reaction(data.get("reaction", "")),
            )
    except (json.JSONDecodeError, TypeError, ValueError):
        pass  # Not JSON, try text format

    # Try text format (DECISION: RESPOND)
    lines = response_text.strip().split('\n')
    text_lower = response_text.lower()

    decision = None
    confidence = 0.5
    reasoning = None
    reaction = None
    found_structured_format = False

    for line in lines:
        line = line.strip()
        if line.startswith("DECISION:"):
            found_structured_format = True
            decision_str = line.replace("DECISION:", "").strip().upper()
            # Handle "RESPOND | ACKNOWLEDGE_ONLY | STAY_QUIET" format
            decision_str = decision_str.split("|")[0].strip()
            if decision_str == "RESPOND":
                decision = ChimeInDecision.RESPOND
            elif decision_str == "ACKNOWLEDGE_ONLY":
                decision = ChimeInDecision.ACKNOWLEDGE_ONLY
            elif decision_str == "NOT_RELEVANT":
                decision = ChimeInDecision.NOT_RELEVANT
            else:
                decision = ChimeInDecision.STAY_QUIET

        elif line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.replace("CONFIDENCE:", "").strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                confidence = 0.5

        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

        elif line.startswith("REACTION:"):
            reaction = line.replace("REACTION:", "").strip()

    if found_structured_format and decision is not None:
        return ChimeInResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            suggested_reaction=_parse_reaction(reaction) if reaction else None,
        )

    # Natural language fallback - default to STAY_QUIET
    # Check for explicit "respond" indicators
    respond_indicators = [
        "i should respond", "i will respond", "i'll respond",
        "i should chime in", "i will chime in", "i'll chime in",
        "this is relevant", "my expertise", "i can help",
    ]

    should_respond = any(indicator in text_lower for indicator in respond_indicators)

    if should_respond:
        return ChimeInResult(
            decision=ChimeInDecision.RESPOND,
            confidence=0.5,
            reasoning="Inferred RESPOND from natural language",
        )

    # Default: STAY_QUIET when format not followed
    return ChimeInResult(
        decision=ChimeInDecision.STAY_QUIET,
        confidence=0.5,
        reasoning="Unstructured response - defaulting to STAY_QUIET",
    )


def clean_mention_tokens(content: str, agent_registry: Dict[str, int]) -> str:
    """
    Replace Discord mention tokens with readable @AgentName format.

    Args:
        content: Message content with raw <@123456789> tokens
        agent_registry: Dict mapping agent names to their Discord user IDs

    Returns:
        Content with mentions replaced by @AgentName
    """
    for name, user_id in agent_registry.items():
        content = content.replace(f"<@{user_id}>", f"@{name.capitalize()}")
        content = content.replace(f"<@!{user_id}>", f"@{name.capitalize()}")
    return content


def build_evaluation_prompt(
    agent_name: str,
    agent_description: str,
    context: ChimeInContext,
    memory_context: str = "",
    mentioned_agents: Optional[List[str]] = None,
) -> str:
    """
    Build the evaluation prompt for chime-in decision.

    Args:
        agent_name: Name of the evaluating agent
        agent_description: Description of agent's domain/personality
        context: ChimeInContext with message details
        memory_context: Relevant memories about chime-in preferences
        mentioned_agents: List of agent names that were @mentioned in the message

    Returns:
        Formatted prompt string
    """
    recent_context = "\n".join(
        f"  - {msg}" for msg in context.recent_messages[-10:]
    ) if context.recent_messages else "(no recent context)"

    # Format mentioned agents for the prompt
    if mentioned_agents:
        mentioned_str = ", ".join(f"@{name}" for name in mentioned_agents)
    else:
        mentioned_str = "(none - message is to the general channel)"

    return CHIME_IN_EVALUATION_PROMPT.format(
        agent_name=agent_name,
        message=context.message_content,
        author=context.message_author,
        recent_context=recent_context,
        memory_context=memory_context or "(no relevant memories)",
        agent_description=agent_description,
        mentioned_agents=mentioned_str,
    )
