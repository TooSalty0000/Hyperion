"""Chime-in evaluation for collaborative agent responses."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


# Structured output schema for chime-in decisions (Gemini)
CHIME_IN_DECISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "decision": {
            "type": "STRING",
            "enum": ["RESPOND", "STAY_QUIET", "NOT_RELEVANT"],
            "description": "Whether to chime in on this message"
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence level from 0.0 to 1.0"
        },
        "reasoning": {
            "type": "STRING",
            "description": "Brief explanation for the decision"
        }
    },
    "required": ["decision", "confidence", "reasoning"]
}


class ChimeInDecision(Enum):
    """Decision about whether to chime in on a conversation."""
    RESPOND = "respond"        # Agent should respond
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

    @property
    def should_respond(self) -> bool:
        """Whether the agent should respond based on this result."""
        return self.decision == ChimeInDecision.RESPOND


# Evaluation prompt template for chime-in decisions (utility LLM - decision only)
CHIME_IN_EVALUATION_PROMPT = """INTERNAL DECISION PROCESS - DO NOT RESPOND TO THE USER

You are the decision module for {agent_name}. Output a decision about whether to chime in.

=== INPUT ===
Message: "{message}"
From: {author}
Your domain: {agent_description}
Recent conversation:
{recent_context}
Memories: {memory_context}

=== MANDATORY RESPOND (ignore domain, these ALWAYS trigger RESPOND) ===
- Words like "everyone", "all of you", "please respond", "say hi" → RESPOND
- Your name mentioned (even without @) → RESPOND
- Direct question to the group → RESPOND

=== STAY_QUIET only when ===
- None of the mandatory triggers above apply AND
- Topic is completely unrelated to you AND
- Someone else already fully handled it

=== OUTPUT FORMAT ===
DECISION: RESPOND
CONFIDENCE: 0.9
REASONING: [reason]

OR

DECISION: STAY_QUIET
CONFIDENCE: 0.9
REASONING: [reason]
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
            elif decision_str == "NOT_RELEVANT":
                decision = ChimeInDecision.NOT_RELEVANT
            else:
                decision = ChimeInDecision.STAY_QUIET

            return ChimeInResult(
                decision=decision,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
    except (json.JSONDecodeError, TypeError, ValueError):
        pass  # Not JSON, try text format

    # Try text format (DECISION: RESPOND)
    lines = response_text.strip().split('\n')
    text_lower = response_text.lower()

    decision = None
    confidence = 0.5
    reasoning = None
    found_structured_format = False

    for line in lines:
        line = line.strip()
        if line.startswith("DECISION:"):
            found_structured_format = True
            decision_str = line.replace("DECISION:", "").strip().upper()
            if decision_str == "RESPOND":
                decision = ChimeInDecision.RESPOND
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

    if found_structured_format and decision is not None:
        return ChimeInResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
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


def build_evaluation_prompt(
    agent_name: str,
    agent_description: str,
    context: ChimeInContext,
    memory_context: str = "",
) -> str:
    """
    Build the evaluation prompt for chime-in decision.

    Args:
        agent_name: Name of the evaluating agent
        agent_description: Description of agent's domain/personality
        context: ChimeInContext with message details
        memory_context: Relevant memories about chime-in preferences

    Returns:
        Formatted prompt string
    """
    recent_context = "\n".join(
        f"  - {msg}" for msg in context.recent_messages[-10:]
    ) if context.recent_messages else "(no recent context)"

    return CHIME_IN_EVALUATION_PROMPT.format(
        agent_name=agent_name,
        message=context.message_content,
        author=context.message_author,
        recent_context=recent_context,
        memory_context=memory_context or "(no relevant memories)",
        agent_description=agent_description,
    )
