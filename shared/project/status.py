"""Project status tracking with pattern-based monitoring."""

import re
import json
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

from .models import ProjectState, ProjectStatus

logger = logging.getLogger(__name__)


class OutputPatternMatcher:
    """
    Pattern-based extraction of status from terminal output.

    Uses regex patterns to detect status changes without LLM calls,
    making it efficient for real-time monitoring.
    """

    # Patterns indicating success/completion
    SUCCESS_PATTERNS = [
        (r'✓|✔|PASS|SUCCESS|COMPLETE|DONE|FINISHED', 10),
        (r'\d+ passing', 5),
        (r'Build succeeded', 10),
        (r'Compilation successful', 10),
        (r'All tests passed', 10),
        (r'\[success\]', 8),
    ]

    # Patterns indicating errors/failures
    ERROR_PATTERNS = [
        (r'✗|✘|FAIL|ERROR|FAILED', 10),
        (r'\d+ failing', 8),
        (r'Build failed', 10),
        (r'Compilation error', 10),
        (r'Tests? failed', 10),
        (r'\[error\]', 8),
        (r'Exception:', 7),
        (r'Traceback', 7),
        (r'Error:', 5),
    ]

    # Patterns indicating work in progress
    PROGRESS_PATTERNS = [
        (r'Running|Compiling|Building|Testing|Installing|Downloading', 5),
        (r'⠋|⠙|⠹|⠸|⠼|⠴|⠦|⠧|⠇|⠏', 3),  # Spinner characters
        (r'\.\.\.$', 2),  # Trailing dots
        (r'\d+%', 5),  # Percentage
        (r'\[\d+/\d+\]', 5),  # Progress like [3/10]
    ]

    # Patterns indicating blockers/waiting
    BLOCKED_PATTERNS = [
        (r'waiting for|blocked by|depends on', 8),
        (r'Connection refused|timeout', 7),
        (r'Permission denied', 8),
        (r'Rate limit', 7),
        (r'quota exceeded', 7),
    ]

    @classmethod
    def analyze_output(cls, output: str) -> Dict[str, Any]:
        """
        Analyze terminal output and extract status indicators.

        Args:
            output: Terminal output text

        Returns:
            Dict with detected status, confidence, and details
        """
        result = {
            "success_score": 0,
            "error_score": 0,
            "progress_score": 0,
            "blocked_score": 0,
            "patterns_matched": [],
            "suggested_status": None,
            "confidence": 0.0,
        }

        # Check last N lines for relevance
        lines = output.strip().split('\n')[-20:]  # Last 20 lines
        recent_output = '\n'.join(lines)

        # Match patterns
        for pattern, weight in cls.SUCCESS_PATTERNS:
            if re.search(pattern, recent_output, re.IGNORECASE):
                result["success_score"] += weight
                result["patterns_matched"].append(("success", pattern))

        for pattern, weight in cls.ERROR_PATTERNS:
            if re.search(pattern, recent_output, re.IGNORECASE):
                result["error_score"] += weight
                result["patterns_matched"].append(("error", pattern))

        for pattern, weight in cls.PROGRESS_PATTERNS:
            if re.search(pattern, recent_output, re.IGNORECASE):
                result["progress_score"] += weight
                result["patterns_matched"].append(("progress", pattern))

        for pattern, weight in cls.BLOCKED_PATTERNS:
            if re.search(pattern, recent_output, re.IGNORECASE):
                result["blocked_score"] += weight
                result["patterns_matched"].append(("blocked", pattern))

        # Determine suggested status
        scores = {
            "error": result["error_score"],
            "blocked": result["blocked_score"],
            "complete": result["success_score"],
            "working": result["progress_score"],
        }

        max_score = max(scores.values())
        if max_score > 0:
            result["suggested_status"] = max(scores, key=scores.get)
            result["confidence"] = min(1.0, max_score / 20.0)

        return result

    @classmethod
    def extract_progress_percent(cls, output: str) -> Optional[int]:
        """
        Extract progress percentage from output if available.

        Args:
            output: Terminal output text

        Returns:
            Progress percentage (0-100) or None
        """
        lines = output.strip().split('\n')[-10:]
        recent = '\n'.join(lines)

        # Look for percentage patterns
        matches = re.findall(r'(\d{1,3})%', recent)
        if matches:
            # Return the last percentage found
            percent = int(matches[-1])
            return min(100, max(0, percent))

        # Look for fraction patterns like [3/10]
        fraction_match = re.search(r'\[(\d+)/(\d+)\]', recent)
        if fraction_match:
            current, total = int(fraction_match.group(1)), int(fraction_match.group(2))
            if total > 0:
                return min(100, int((current / total) * 100))

        return None

    @classmethod
    def extract_summary(cls, output: str, max_length: int = 200) -> str:
        """
        Extract a brief summary from the output.

        Args:
            output: Terminal output text
            max_length: Maximum summary length

        Returns:
            Brief summary string
        """
        lines = output.strip().split('\n')

        # Look for summary-like lines at the end
        for line in reversed(lines[-10:]):
            line = line.strip()
            # Skip empty lines and pure progress indicators
            if not line or re.match(r'^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏\s]+$', line):
                continue
            # Skip lines that are just separators
            if re.match(r'^[-=_\s]+$', line):
                continue
            # Found a meaningful line
            if len(line) <= max_length:
                return line
            return line[:max_length-3] + "..."

        return "No summary available"


class ProjectStatusTracker:
    """
    Tracks and persists project status using Redis.

    Provides pattern-based monitoring of terminal output
    and maintains status history for each project.
    """

    KEY_PREFIX = "vega:project_state:"

    def __init__(self, redis_url: str):
        """
        Initialize the status tracker.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._redis = None

    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.redis_url)
        return self._redis

    async def get_state(self, project_name: str) -> Optional[ProjectState]:
        """
        Get the current state of a project.

        Args:
            project_name: Project name

        Returns:
            ProjectState or None if not found
        """
        redis = await self._get_redis()
        key = f"{self.KEY_PREFIX}{project_name}"

        data = await redis.get(key)
        if data:
            return ProjectState.from_dict(json.loads(data))
        return None

    async def get_or_create_state(self, project_name: str) -> ProjectState:
        """
        Get or create state for a project.

        Args:
            project_name: Project name

        Returns:
            ProjectState (existing or new)
        """
        state = await self.get_state(project_name)
        if state is None:
            state = ProjectState(project_name=project_name)
            await self.save_state(state)
        return state

    async def save_state(self, state: ProjectState) -> None:
        """
        Save project state to Redis.

        Args:
            state: ProjectState to save
        """
        redis = await self._get_redis()
        key = f"{self.KEY_PREFIX}{state.project_name}"

        await redis.set(key, json.dumps(state.to_dict()))

    async def update_status(
        self,
        project_name: str,
        status: ProjectStatus,
        message: str = "",
        session_id: int = None
    ) -> ProjectState:
        """
        Update the status of a project.

        Args:
            project_name: Project name
            status: New status
            message: Status message
            session_id: Associated session ID

        Returns:
            Updated ProjectState
        """
        state = await self.get_or_create_state(project_name)
        state.update_status(status, message, session_id)
        await self.save_state(state)
        return state

    async def update_from_output(
        self,
        project_name: str,
        output: str,
        session_id: int = None
    ) -> ProjectState:
        """
        Update project status based on terminal output analysis.

        Uses pattern matching to detect status changes.

        Args:
            project_name: Project name
            output: Terminal output to analyze
            session_id: Associated session ID

        Returns:
            Updated ProjectState
        """
        state = await self.get_or_create_state(project_name)

        # Analyze output
        analysis = OutputPatternMatcher.analyze_output(output)

        # Update state if we have a confident suggestion
        if analysis["suggested_status"] and analysis["confidence"] >= 0.3:
            suggested = analysis["suggested_status"]

            # Map to ProjectStatus
            status_map = {
                "error": ProjectStatus.ERROR,
                "blocked": ProjectStatus.BLOCKED,
                "complete": ProjectStatus.COMPLETE,
                "working": ProjectStatus.WORKING,
            }

            new_status = status_map.get(suggested, ProjectStatus.WORKING)

            # Only update if status changed
            if new_status != state.status:
                state.update_status(
                    new_status,
                    f"Auto-detected: {analysis['patterns_matched'][:3]}",
                    session_id
                )

        # Update progress if available
        progress = OutputPatternMatcher.extract_progress_percent(output)
        if progress is not None:
            state.progress_percent = progress

        # Update summary
        state.last_output_summary = OutputPatternMatcher.extract_summary(output)
        state.current_session_id = session_id

        await self.save_state(state)
        return state

    async def set_task(
        self,
        project_name: str,
        task: str,
        session_id: int = None
    ) -> ProjectState:
        """
        Set the current task for a project.

        Args:
            project_name: Project name
            task: Task description
            session_id: Associated session ID

        Returns:
            Updated ProjectState
        """
        state = await self.get_or_create_state(project_name)
        state.current_task = task
        state.current_session_id = session_id
        state.update_status(ProjectStatus.WORKING, f"Started: {task[:50]}...", session_id)
        await self.save_state(state)
        return state

    async def complete_task(
        self,
        project_name: str,
        message: str = "Task completed"
    ) -> ProjectState:
        """
        Mark the current task as complete.

        Args:
            project_name: Project name
            message: Completion message

        Returns:
            Updated ProjectState
        """
        state = await self.get_or_create_state(project_name)
        state.current_task = None
        state.progress_percent = 100
        state.update_status(ProjectStatus.COMPLETE, message, state.current_session_id)
        await self.save_state(state)
        return state

    async def add_blocker(
        self,
        project_name: str,
        blocker: str
    ) -> ProjectState:
        """
        Add a blocker to a project.

        Args:
            project_name: Project name
            blocker: Blocker description

        Returns:
            Updated ProjectState
        """
        state = await self.get_or_create_state(project_name)
        state.add_blocker(blocker)
        state.update_status(ProjectStatus.BLOCKED, f"Blocked: {blocker}", state.current_session_id)
        await self.save_state(state)
        return state

    async def clear_blockers(self, project_name: str) -> ProjectState:
        """
        Clear all blockers for a project.

        Args:
            project_name: Project name

        Returns:
            Updated ProjectState
        """
        state = await self.get_or_create_state(project_name)
        state.clear_blockers()
        await self.save_state(state)
        return state

    async def list_all_states(self) -> List[ProjectState]:
        """
        List all project states.

        Returns:
            List of ProjectState objects
        """
        redis = await self._get_redis()
        keys = await redis.keys(f"{self.KEY_PREFIX}*")

        states = []
        for key in keys:
            data = await redis.get(key)
            if data:
                states.append(ProjectState.from_dict(json.loads(data)))

        return sorted(states, key=lambda s: s.updated_at, reverse=True)

    async def delete_state(self, project_name: str) -> bool:
        """
        Delete state for a project.

        Args:
            project_name: Project name

        Returns:
            True if deleted, False if not found
        """
        redis = await self._get_redis()
        key = f"{self.KEY_PREFIX}{project_name}"

        result = await redis.delete(key)
        return result > 0
