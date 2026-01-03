"""Navigation scratchpad for Canopus working memory."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Finding:
    """A single piece of discovered information."""
    key: str
    value: Any
    url: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Optional[str] = None


@dataclass
class NavigationScratchpad:
    """
    Working memory for a single navigation request.

    Provides structured storage for:
    - Goal and plan steps
    - Progress tracking
    - Discovered findings
    - URL history
    - Goal completion status
    """
    goal: str = ""
    plan_steps: List[str] = field(default_factory=list)
    current_step_index: int = 0
    findings: List[Finding] = field(default_factory=list)
    visited_urls: List[str] = field(default_factory=list)
    goal_achieved: bool = False
    completion_reason: str = ""

    def set_goal(self, goal: str) -> None:
        """Set the navigation goal."""
        self.goal = goal

    def set_plan(self, steps: List[str]) -> None:
        """Set the plan steps."""
        self.plan_steps = steps
        self.current_step_index = 0

    def get_current_step(self) -> Optional[str]:
        """Get the current plan step."""
        if not self.plan_steps:
            return None
        if self.current_step_index >= len(self.plan_steps):
            return None
        return self.plan_steps[self.current_step_index]

    def get_remaining_steps(self) -> List[str]:
        """Get remaining plan steps."""
        if not self.plan_steps:
            return []
        return self.plan_steps[self.current_step_index:]

    def get_completed_steps(self) -> List[str]:
        """Get completed plan steps."""
        if not self.plan_steps:
            return []
        return self.plan_steps[:self.current_step_index]

    def mark_step_complete(self) -> Optional[str]:
        """
        Mark current step as complete and advance to next.

        Returns:
            The next step, or None if all steps complete.
        """
        if self.plan_steps and self.current_step_index < len(self.plan_steps):
            self.current_step_index += 1
        return self.get_current_step()

    def add_finding(
        self,
        key: str,
        value: Any,
        url: str = "",
        context: Optional[str] = None
    ) -> None:
        """Store a discovered piece of information."""
        self.findings.append(Finding(
            key=key,
            value=value,
            url=url,
            context=context
        ))

    def get_finding(self, key: str) -> Optional[Finding]:
        """Get a specific finding by key."""
        for finding in self.findings:
            if finding.key == key:
                return finding
        return None

    def get_all_findings(self) -> List[Finding]:
        """Get all stored findings."""
        return self.findings

    def get_findings_formatted(self) -> str:
        """Get all findings as formatted string."""
        if not self.findings:
            return "No findings stored yet."

        lines = ["=== Stored Findings ==="]
        for i, f in enumerate(self.findings, 1):
            lines.append(f"\n[{i}] {f.key}:")
            lines.append(f"    Value: {f.value}")
            if f.url:
                lines.append(f"    URL: {f.url}")
            if f.context:
                lines.append(f"    Context: {f.context}")

        return "\n".join(lines)

    def add_visited_url(self, url: str) -> None:
        """Record a visited URL."""
        if url and url not in self.visited_urls:
            self.visited_urls.append(url)

    def set_goal_achieved(self, achieved: bool, reason: str = "") -> None:
        """Mark the goal as achieved or not."""
        self.goal_achieved = achieved
        self.completion_reason = reason

    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        lines = ["=== Navigation Progress ==="]

        # Goal
        lines.append(f"\nGoal: {self.goal or 'Not set'}")

        # Plan progress
        if self.plan_steps:
            completed = self.current_step_index
            total = len(self.plan_steps)
            lines.append(f"\nPlan Progress: {completed}/{total} steps")

            if self.get_completed_steps():
                lines.append("\nCompleted:")
                for step in self.get_completed_steps():
                    lines.append(f"  ✓ {step}")

            current = self.get_current_step()
            if current:
                lines.append(f"\nCurrent: → {current}")

            remaining = self.get_remaining_steps()[1:] if len(self.get_remaining_steps()) > 1 else []
            if remaining:
                lines.append("\nRemaining:")
                for step in remaining:
                    lines.append(f"  ○ {step}")
        else:
            lines.append("\nNo plan created yet.")

        # Findings count
        lines.append(f"\nFindings: {len(self.findings)} items stored")

        # URLs visited
        lines.append(f"URLs visited: {len(self.visited_urls)}")

        # Goal status
        if self.goal_achieved:
            lines.append(f"\n✅ Goal Achieved: {self.completion_reason}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset the scratchpad for a new navigation task."""
        self.goal = ""
        self.plan_steps = []
        self.current_step_index = 0
        self.findings = []
        self.visited_urls = []
        self.goal_achieved = False
        self.completion_reason = ""
