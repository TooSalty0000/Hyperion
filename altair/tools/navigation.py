"""Read-only filesystem navigation tools for Altair.

These tools allow Altair to explore the filesystem without modifying anything.
For actual code changes, Altair should use Claude Code via sessions.
"""

import os
import stat
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


class ListDirectoryTool(Tool):
    """List contents of a directory (like ls)."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return (
            "List the contents of a directory. Shows files and subdirectories "
            "with their types and sizes. Use this to explore project structures "
            "or find files before working with them."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Directory path to list. Defaults to current working directory.",
                required=False,
            ),
            ToolParameter(
                name="show_hidden",
                type="boolean",
                description="Include hidden files (starting with dot). Default: False",
                required=False,
            ),
            ToolParameter(
                name="recursive",
                type="boolean",
                description="List subdirectories recursively (max 2 levels deep). Default: False",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        path: Optional[str] = None,
        show_hidden: bool = False,
        recursive: bool = False,
    ) -> str:
        try:
            target_path = Path(path) if path else Path.cwd()

            if not target_path.exists():
                return f"Error: Path does not exist: {target_path}"

            if not target_path.is_dir():
                return f"Error: Not a directory: {target_path}"

            return self._list_dir(target_path, show_hidden, recursive, depth=0, max_depth=2)

        except PermissionError:
            return f"Error: Permission denied accessing {path}"
        except Exception as e:
            logger.error(f"Error listing directory: {e}", exc_info=True)
            return f"Error: {str(e)}"

    def _list_dir(
        self,
        path: Path,
        show_hidden: bool,
        recursive: bool,
        depth: int,
        max_depth: int
    ) -> str:
        indent = "  " * depth
        lines = []

        if depth == 0:
            lines.append(f"📁 {path}/")

        try:
            entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return f"{indent}  (permission denied)"

        for entry in entries:
            # Skip hidden files unless requested
            if entry.name.startswith('.') and not show_hidden:
                continue

            try:
                if entry.is_dir():
                    lines.append(f"{indent}  📁 {entry.name}/")
                    if recursive and depth < max_depth:
                        sub_content = self._list_dir(
                            entry, show_hidden, recursive, depth + 1, max_depth
                        )
                        if sub_content:
                            lines.append(sub_content)
                else:
                    size = self._format_size(entry.stat().st_size)
                    icon = self._get_file_icon(entry.name)
                    lines.append(f"{indent}  {icon} {entry.name} ({size})")
            except (PermissionError, OSError):
                lines.append(f"{indent}  ⚠️ {entry.name} (access error)")

        return "\n".join(lines)

    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}" if unit != 'B' else f"{size}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    def _get_file_icon(self, name: str) -> str:
        ext = Path(name).suffix.lower()
        icons = {
            '.py': '🐍', '.js': '📜', '.ts': '📘', '.jsx': '⚛️', '.tsx': '⚛️',
            '.json': '📋', '.yml': '📋', '.yaml': '📋', '.toml': '📋',
            '.md': '📝', '.txt': '📄', '.rst': '📄',
            '.html': '🌐', '.css': '🎨', '.scss': '🎨',
            '.sh': '💻', '.bash': '💻', '.zsh': '💻',
            '.png': '🖼️', '.jpg': '🖼️', '.gif': '🖼️', '.svg': '🖼️',
            '.lock': '🔒', '.env': '🔐',
        }
        return icons.get(ext, '📄')


class GetCurrentDirectoryTool(Tool):
    """Get the current working directory (like pwd)."""

    @property
    def name(self) -> str:
        return "get_current_directory"

    @property
    def description(self) -> str:
        return "Get the current working directory path."

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        return f"Current directory: {os.getcwd()}"


class ReadFileTool(Tool):
    """Read the contents of a file (read-only)."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. Use this to examine code, config files, "
            "or documentation. This is READ-ONLY - to modify files, use Claude Code "
            "in a session."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file to read",
                required=True,
            ),
            ToolParameter(
                name="lines",
                type="integer",
                description="Maximum number of lines to read. Default: 100",
                required=False,
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="Line number to start reading from (1-indexed). Default: 1",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        path: str,
        lines: int = 100,
        start_line: int = 1,
    ) -> str:
        try:
            file_path = Path(path)

            if not file_path.exists():
                return f"Error: File does not exist: {path}"

            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            # Check file size to avoid reading huge files
            file_size = file_path.stat().st_size
            if file_size > 1_000_000:  # 1MB limit
                return f"Error: File too large ({self._format_size(file_size)}). Use Claude Code for large files."

            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                all_lines = f.readlines()

            total_lines = len(all_lines)

            # Adjust start_line to 0-indexed
            start_idx = max(0, start_line - 1)
            end_idx = min(start_idx + lines, total_lines)

            selected_lines = all_lines[start_idx:end_idx]

            # Format with line numbers
            numbered_lines = []
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                numbered_lines.append(f"{i:4d} | {line.rstrip()}")

            content = "\n".join(numbered_lines)

            header = f"📄 {file_path.name} (lines {start_idx + 1}-{end_idx} of {total_lines})"
            if end_idx < total_lines:
                header += f"\n(Use start_line={end_idx + 1} to continue reading)"

            return f"{header}\n{'─' * 50}\n{content}"

        except PermissionError:
            return f"Error: Permission denied reading {path}"
        except UnicodeDecodeError:
            return f"Error: Cannot read binary file: {path}"
        except Exception as e:
            logger.error(f"Error reading file: {e}", exc_info=True)
            return f"Error: {str(e)}"

    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}" if unit != 'B' else f"{size}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


class FindFilesTool(Tool):
    """Find files matching a pattern (like find/glob)."""

    @property
    def name(self) -> str:
        return "find_files"

    @property
    def description(self) -> str:
        return (
            "Find files matching a pattern in a directory. Supports glob patterns "
            "like '*.py', '**/*.js', or 'src/**/*.tsx'. Useful for discovering "
            "project structure or finding specific file types."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="pattern",
                type="string",
                description="Glob pattern to match (e.g., '*.py', 'src/**/*.js')",
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Directory to search in. Defaults to current directory.",
                required=False,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results to return. Default: 50",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        pattern: str,
        path: Optional[str] = None,
        max_results: int = 50,
    ) -> str:
        try:
            base_path = Path(path) if path else Path.cwd()

            if not base_path.exists():
                return f"Error: Path does not exist: {base_path}"

            if not base_path.is_dir():
                return f"Error: Not a directory: {base_path}"

            # Find matching files
            matches = list(base_path.glob(pattern))

            # Sort by path
            matches.sort(key=lambda x: str(x).lower())

            total_found = len(matches)
            matches = matches[:max_results]

            if not matches:
                return f"No files matching '{pattern}' found in {base_path}"

            lines = [f"Found {total_found} file(s) matching '{pattern}'"]
            if total_found > max_results:
                lines.append(f"(showing first {max_results})")
            lines.append("")

            for match in matches:
                try:
                    rel_path = match.relative_to(base_path)
                    if match.is_dir():
                        lines.append(f"📁 {rel_path}/")
                    else:
                        size = self._format_size(match.stat().st_size)
                        lines.append(f"📄 {rel_path} ({size})")
                except (ValueError, OSError):
                    lines.append(f"📄 {match}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error finding files: {e}", exc_info=True)
            return f"Error: {str(e)}"

    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}" if unit != 'B' else f"{size}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


class FileInfoTool(Tool):
    """Get detailed information about a file or directory."""

    @property
    def name(self) -> str:
        return "file_info"

    @property
    def description(self) -> str:
        return (
            "Get detailed information about a file or directory including "
            "size, permissions, modification time, and type."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="Path to the file or directory",
                required=True,
            ),
        ]

    async def execute(self, context: ToolContext, path: str) -> str:
        try:
            target = Path(path)

            if not target.exists():
                return f"Error: Path does not exist: {path}"

            stat_info = target.stat()

            # Basic info
            info_lines = [f"📋 Info for: {target}"]
            info_lines.append(f"Type: {'Directory' if target.is_dir() else 'File'}")

            # Size
            if target.is_file():
                info_lines.append(f"Size: {self._format_size(stat_info.st_size)}")
            elif target.is_dir():
                # Count items in directory
                try:
                    items = list(target.iterdir())
                    info_lines.append(f"Contains: {len(items)} items")
                except PermissionError:
                    info_lines.append("Contains: (permission denied)")

            # Times
            mtime = datetime.fromtimestamp(stat_info.st_mtime)
            info_lines.append(f"Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

            # Permissions (simplified)
            mode = stat_info.st_mode
            perms = []
            if mode & stat.S_IRUSR:
                perms.append("read")
            if mode & stat.S_IWUSR:
                perms.append("write")
            if mode & stat.S_IXUSR:
                perms.append("execute")
            info_lines.append(f"Permissions: {', '.join(perms) if perms else 'none'}")

            # For files, show extension info
            if target.is_file():
                ext = target.suffix.lower()
                if ext:
                    info_lines.append(f"Extension: {ext}")

            return "\n".join(info_lines)

        except PermissionError:
            return f"Error: Permission denied accessing {path}"
        except Exception as e:
            logger.error(f"Error getting file info: {e}", exc_info=True)
            return f"Error: {str(e)}"

    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}" if unit != 'B' else f"{size}{unit}"
            size /= 1024
        return f"{size:.1f}TB"
