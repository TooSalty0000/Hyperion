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
    SetSessionNoteTool,
    GetSessionNoteTool,
)
from .tunnel import (
    StartTunnelTool,
    StopTunnelTool,
    ListTunnelsTool,
)
from .navigation import (
    ListDirectoryTool,
    GetCurrentDirectoryTool,
    ReadFileTool,
    FindFilesTool,
    FileInfoTool,
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
    "SetSessionNoteTool",
    "GetSessionNoteTool",
    # Tunnel management (ngrok)
    "StartTunnelTool",
    "StopTunnelTool",
    "ListTunnelsTool",
    # Navigation (read-only)
    "ListDirectoryTool",
    "GetCurrentDirectoryTool",
    "ReadFileTool",
    "FindFilesTool",
    "FileInfoTool",
]
