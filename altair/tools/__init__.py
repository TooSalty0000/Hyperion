"""Altair's tools for CLI management and project execution."""

from .cli_commands import (
    StartClaudeCodeTool,
    SendCLICommandTool,
    GetCLIOutputTool,
    WaitForCompletionTool,
    GetSessionStatusTool,
    ListActiveSessionsTool,
)
from .session import (
    CreateSessionTool,
    TerminateSessionTool,
    SwitchSessionTool,
)

__all__ = [
    # CLI commands
    "StartClaudeCodeTool",
    "SendCLICommandTool",
    "GetCLIOutputTool",
    "WaitForCompletionTool",
    "GetSessionStatusTool",
    "ListActiveSessionsTool",
    # Session management
    "CreateSessionTool",
    "TerminateSessionTool",
    "SwitchSessionTool",
]
