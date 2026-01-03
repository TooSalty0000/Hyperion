"""Workflow state management for agent context awareness."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Callable, Any
from abc import ABC, abstractmethod


@dataclass
class WorkflowState:
    """
    Current state of an agent's workflow.

    This is injected into the LLM's context so it can make informed
    decisions about how to handle new messages relative to ongoing work.
    """
    # Current task (if any)
    current_task: Optional[str] = None
    current_task_started: Optional[datetime] = None
    current_task_tools_used: List[str] = field(default_factory=list)
    current_task_progress: List[str] = field(default_factory=list)

    # Last completed task (for continuity between tasks)
    last_completed_task: Optional[str] = None
    last_completed_at: Optional[datetime] = None

    # Queue state
    pending_messages: int = 0

    # Agent identity
    agent_name: str = ""

    @property
    def is_busy(self) -> bool:
        """Check if currently processing a task."""
        return self.current_task is not None

    @property
    def current_task_elapsed(self) -> int:
        """Seconds since current task started."""
        if self.current_task_started:
            return int((datetime.now() - self.current_task_started).total_seconds())
        return 0

    @property
    def time_since_last_task(self) -> Optional[int]:
        """Seconds since last task completed."""
        if self.last_completed_at:
            return int((datetime.now() - self.last_completed_at).total_seconds())
        return None

    def format_for_llm(self) -> str:
        """
        Format workflow state for injection into LLM context.

        This provides the LLM with awareness of:
        - What it's currently working on
        - How long it's been working
        - What tools it has used
        - What it recently completed
        - What's waiting in queue
        """
        lines = ["\n## CURRENT WORKFLOW STATE"]

        if self.current_task:
            lines.append(f"**Status:** BUSY - Processing a task")
            lines.append(f"**Current Task:** {self.current_task}")
            lines.append(f"**Working for:** {self.current_task_elapsed} seconds")

            if self.current_task_tools_used:
                recent_tools = self.current_task_tools_used[-5:]  # Last 5
                lines.append(f"**Tools used:** {', '.join(recent_tools)}")

            if self.current_task_progress:
                lines.append(f"**Latest progress:** {self.current_task_progress[-1]}")
        else:
            lines.append(f"**Status:** IDLE - Ready for new tasks")

        if self.last_completed_task and self.time_since_last_task:
            if self.time_since_last_task < 300:  # Within 5 minutes
                lines.append(f"**Just completed ({self.time_since_last_task}s ago):** {self.last_completed_task[:100]}")

        if self.pending_messages > 0:
            lines.append(f"**Messages waiting in queue:** {self.pending_messages}")

        lines.append("")
        lines.append("Use this state to understand if incoming messages relate to your current/recent work.")
        lines.append("If someone asks for something you're already doing, acknowledge it rather than starting over.")
        lines.append("If they add new requirements, incorporate them into your current work.")

        return "\n".join(lines)


class WorkflowStateProvider(ABC):
    """Abstract interface for providing workflow state to agents."""

    @abstractmethod
    def get_workflow_state(self) -> WorkflowState:
        """Get the current workflow state."""
        pass

    @abstractmethod
    def update_progress(self, status: str):
        """Update the current task's progress."""
        pass

    @abstractmethod
    def record_tool_use(self, tool_name: str):
        """Record that a tool was used."""
        pass


class QueueWorkflowStateProvider(WorkflowStateProvider):
    """
    Workflow state provider backed by AgentMessageQueue.

    Bridges the queue's TaskState to the agent's WorkflowState.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._queue = None  # Set by cog after queue is created
        self._last_completed_task: Optional[str] = None
        self._last_completed_at: Optional[datetime] = None

    def set_queue(self, queue):
        """Connect to the message queue."""
        self._queue = queue

    def get_workflow_state(self) -> WorkflowState:
        """Build workflow state from queue's task state."""
        state = WorkflowState(agent_name=self.agent_name)

        if self._queue:
            task = self._queue.current_task
            if task:
                state.current_task = task.content
                state.current_task_started = task.started_at
                state.current_task_tools_used = task.tools_used.copy()
                state.current_task_progress = task.status_updates.copy()

            state.pending_messages = self._queue.queue_size

        state.last_completed_task = self._last_completed_task
        state.last_completed_at = self._last_completed_at

        return state

    def update_progress(self, status: str):
        """Update progress via the queue."""
        if self._queue:
            self._queue.update_task_status(status)

    def record_tool_use(self, tool_name: str):
        """Record tool use via the queue."""
        if self._queue:
            self._queue.record_tool_use(tool_name)

    def on_task_completed(self, task_content: str):
        """Called when a task completes (for continuity tracking)."""
        self._last_completed_task = task_content
        self._last_completed_at = datetime.now()
