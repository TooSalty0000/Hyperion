"""Agent control tools for self-management and task flow control."""

import logging
from typing import Optional, List

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class AbortTaskTool(Tool):
    """
    Abort the current task and return to the user for clarification.

    Use this tool when:
    - You realize you need more information to proceed
    - The user's request is ambiguous and you need clarification
    - You encounter a situation where continuing would be wrong
    - You want to confirm an important assumption before proceeding

    When called, this immediately stops the current processing loop
    and returns your message to the user.
    """

    # Special marker that the agent loop checks for
    ABORT_MARKER = "[ABORT_TASK]"

    @property
    def name(self) -> str:
        return "abort_task"

    @property
    def description(self) -> str:
        return (
            "Stop the current task and ask the user for clarification or more information. "
            "Use this when you realize you can't proceed without additional input, "
            "or when the request is ambiguous and you need to confirm before continuing. "
            "Your message will be shown to the user immediately."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="reason",
                type="string",
                description=(
                    "Explain why you're stopping and what information you need. "
                    "This message will be sent to the user. Be specific about what "
                    "clarification or information would help you proceed."
                ),
                required=True
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        reason: str,
    ) -> str:
        """Abort the task and return reason to user."""
        logger.info(f"Task aborted by agent: {reason[:100]}...")

        # Return special marker that the agent loop detects
        return f"{self.ABORT_MARKER}\n{reason}"


class RequestClarificationTool(Tool):
    """
    Ask the user a specific question without fully aborting the task.

    Use this for quick clarifications where you want to pause and wait
    for an answer before continuing with the current task.
    """

    CLARIFICATION_MARKER = "[NEEDS_CLARIFICATION]"

    @property
    def name(self) -> str:
        return "request_clarification"

    @property
    def description(self) -> str:
        return (
            "Ask the user a specific question and wait for their response. "
            "Use this when you need a quick yes/no or choice answer to proceed. "
            "The task will pause until the user responds."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="question",
                type="string",
                description="The specific question to ask the user. Be clear and concise.",
                required=True
            ),
            ToolParameter(
                name="options",
                type="string",
                description=(
                    "Optional: comma-separated list of options for the user to choose from. "
                    "Example: 'yes, no' or 'option A, option B, option C'"
                ),
                required=False
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        question: str,
        options: Optional[str] = None,
    ) -> str:
        """Request clarification from user."""
        logger.info(f"Requesting clarification: {question[:100]}...")

        if options:
            return f"{self.CLARIFICATION_MARKER}\n**Question:** {question}\n**Options:** {options}"
        else:
            return f"{self.CLARIFICATION_MARKER}\n**Question:** {question}"
