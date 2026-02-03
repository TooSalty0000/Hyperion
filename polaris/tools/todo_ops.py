"""Todo operation tools for Polaris."""

import logging
from datetime import datetime
from typing import List, Optional

from shared.tools.interface import Tool
from shared.llm.types import ToolParameter
from polaris.tools.utils import PolarisToolContext
from polaris.todo.models import TodoStatus, TodoPriority

logger = logging.getLogger(__name__)


class CreateTodoTool(Tool):
    """Tool for creating a new todo item."""

    @property
    def name(self) -> str:
        return "create_todo"

    @property
    def description(self) -> str:
        return (
            "Create a new todo item. Specify title, and optionally priority, "
            "due date, tags, description, estimated duration, and project."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="title",
                type="string",
                description="Todo title/summary",
                required=True,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="Detailed description (optional)",
                required=False,
            ),
            ToolParameter(
                name="priority",
                type="string",
                description="Priority level: low, medium, high, critical (default: medium)",
                required=False,
                enum=["low", "medium", "high", "critical"],
            ),
            ToolParameter(
                name="due_date",
                type="string",
                description="Due date (e.g., 'tomorrow', '2025-02-15', 'next Monday')",
                required=False,
            ),
            ToolParameter(
                name="due_time",
                type="string",
                description="Due time (e.g., '5pm', '17:00'). Only used if due_date is set.",
                required=False,
            ),
            ToolParameter(
                name="tags",
                type="string",
                description="Comma-separated tags (e.g., 'work,urgent,pr-review')",
                required=False,
            ),
            ToolParameter(
                name="project",
                type="string",
                description="Project name to group this todo under (optional)",
                required=False,
            ),
            ToolParameter(
                name="estimated_minutes",
                type="integer",
                description="Estimated duration in minutes (optional)",
                required=False,
            ),
            ToolParameter(
                name="parent_todo_id",
                type="string",
                description="Parent todo ID to create this as a subtask (optional)",
                required=False,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.todo_manager:
            return "Error: Todo system not available. Redis is required."

        title = kwargs.get("title")
        if not title:
            return "Error: Title is required."

        # Parse priority
        priority_map = {
            "low": TodoPriority.LOW,
            "medium": TodoPriority.MEDIUM,
            "high": TodoPriority.HIGH,
            "critical": TodoPriority.CRITICAL,
        }
        priority_str = kwargs.get("priority", "medium").lower()
        priority = priority_map.get(priority_str, TodoPriority.MEDIUM)

        # Parse due date
        due_at = None
        if kwargs.get("due_date"):
            try:
                from polaris.tools.utils import parse_datetime
                tz = context.config.timezone if context.config else "UTC"
                due_at = parse_datetime(
                    kwargs["due_date"],
                    kwargs.get("due_time"),
                    timezone=tz,
                )
                # Strip tzinfo for UTC storage
                if due_at.tzinfo:
                    due_at = due_at.replace(tzinfo=None)
            except ValueError as e:
                return f"Error parsing due date: {e}"

        # Parse tags
        tags = []
        if kwargs.get("tags"):
            tags = [t.strip() for t in kwargs["tags"].split(",") if t.strip()]

        user_id = str(context.user_id) if context.user_id else "default"

        try:
            todo = await context.todo_manager.create(
                user_id=user_id,
                title=title,
                description=kwargs.get("description"),
                priority=priority,
                tags=tags,
                project_id=kwargs.get("project"),
                parent_id=kwargs.get("parent_todo_id"),
                due_at=due_at,
                estimated_duration_minutes=kwargs.get("estimated_minutes"),
            )

            response = f"Todo created: **{todo.title}**\n"
            response += f"- ID: {todo.id}\n"
            response += f"- Priority: {priority_str}\n"
            if due_at:
                response += f"- Due: {due_at.strftime('%B %d, %Y %I:%M %p')}\n"
            if tags:
                response += f"- Tags: {', '.join(tags)}\n"
            if kwargs.get("project"):
                response += f"- Project: {kwargs['project']}\n"

            return response

        except Exception as e:
            logger.error(f"Error creating todo: {e}", exc_info=True)
            return f"Error creating todo: {e}"


class ListTodosTool(Tool):
    """Tool for listing todos with filters."""

    @property
    def name(self) -> str:
        return "list_todos"

    @property
    def description(self) -> str:
        return (
            "List todo items with optional filters. Can filter by status, "
            "priority, project, tag, or due date range."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="status",
                type="string",
                description="Filter by status: pending, in_progress, completed, cancelled, deferred",
                required=False,
                enum=["pending", "in_progress", "completed", "cancelled", "deferred"],
            ),
            ToolParameter(
                name="priority",
                type="string",
                description="Filter by priority: low, medium, high, critical",
                required=False,
                enum=["low", "medium", "high", "critical"],
            ),
            ToolParameter(
                name="project",
                type="string",
                description="Filter by project name",
                required=False,
            ),
            ToolParameter(
                name="tag",
                type="string",
                description="Filter by tag",
                required=False,
            ),
            ToolParameter(
                name="sort_by",
                type="string",
                description="Sort order: priority (default), due, created",
                required=False,
                enum=["priority", "due", "created"],
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Max results (default: 20)",
                required=False,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.todo_manager:
            return "Error: Todo system not available. Redis is required."

        user_id = str(context.user_id) if context.user_id else "default"

        # Parse filters
        status = None
        if kwargs.get("status"):
            try:
                status = TodoStatus(kwargs["status"])
            except ValueError:
                return f"Invalid status: {kwargs['status']}"

        priority = None
        if kwargs.get("priority"):
            priority_map = {
                "low": TodoPriority.LOW,
                "medium": TodoPriority.MEDIUM,
                "high": TodoPriority.HIGH,
                "critical": TodoPriority.CRITICAL,
            }
            priority = priority_map.get(kwargs["priority"].lower())

        try:
            todos = await context.todo_manager.list_todos(
                user_id=user_id,
                status=status,
                priority=priority,
                project_id=kwargs.get("project"),
                tag=kwargs.get("tag"),
                sort_by=kwargs.get("sort_by", "priority"),
                limit=kwargs.get("limit", 20),
            )

            if not todos:
                filters = []
                if status:
                    filters.append(f"status={status.value}")
                if priority:
                    filters.append(f"priority={priority.name.lower()}")
                if kwargs.get("project"):
                    filters.append(f"project={kwargs['project']}")
                if kwargs.get("tag"):
                    filters.append(f"tag={kwargs['tag']}")
                filter_str = f" (filters: {', '.join(filters)})" if filters else ""
                return f"No todos found{filter_str}."

            lines = [f"**Todos** ({len(todos)} items)\n"]
            for todo in todos:
                lines.append(todo.format_for_display())
                lines.append("")  # spacing

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error listing todos: {e}", exc_info=True)
            return f"Error listing todos: {e}"


class UpdateTodoTool(Tool):
    """Tool for updating an existing todo."""

    @property
    def name(self) -> str:
        return "update_todo"

    @property
    def description(self) -> str:
        return (
            "Update an existing todo item. Specify the todo ID and any fields to change. "
            "Use this to change title, description, priority, due date, tags, or status."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="todo_id",
                type="string",
                description="The todo ID to update (e.g., 'todo-1')",
                required=True,
            ),
            ToolParameter(
                name="title",
                type="string",
                description="New title (optional)",
                required=False,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="New description (optional)",
                required=False,
            ),
            ToolParameter(
                name="status",
                type="string",
                description="New status: pending, in_progress, completed, cancelled, deferred",
                required=False,
                enum=["pending", "in_progress", "completed", "cancelled", "deferred"],
            ),
            ToolParameter(
                name="priority",
                type="string",
                description="New priority: low, medium, high, critical",
                required=False,
                enum=["low", "medium", "high", "critical"],
            ),
            ToolParameter(
                name="due_date",
                type="string",
                description="New due date (e.g., 'tomorrow', '2025-02-15')",
                required=False,
            ),
            ToolParameter(
                name="due_time",
                type="string",
                description="New due time (e.g., '5pm', '17:00')",
                required=False,
            ),
            ToolParameter(
                name="tags",
                type="string",
                description="New tags (comma-separated, replaces existing)",
                required=False,
            ),
            ToolParameter(
                name="project",
                type="string",
                description="New project name",
                required=False,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.todo_manager:
            return "Error: Todo system not available. Redis is required."

        todo_id = kwargs.get("todo_id")
        if not todo_id:
            return "Error: todo_id is required."

        # Build update kwargs
        update_kwargs = {}

        if kwargs.get("title"):
            update_kwargs["title"] = kwargs["title"]
        if kwargs.get("description"):
            update_kwargs["description"] = kwargs["description"]
        if kwargs.get("status"):
            try:
                update_kwargs["status"] = TodoStatus(kwargs["status"])
            except ValueError:
                return f"Invalid status: {kwargs['status']}"
        if kwargs.get("priority"):
            priority_map = {
                "low": TodoPriority.LOW,
                "medium": TodoPriority.MEDIUM,
                "high": TodoPriority.HIGH,
                "critical": TodoPriority.CRITICAL,
            }
            p = priority_map.get(kwargs["priority"].lower())
            if p:
                update_kwargs["priority"] = p
        if kwargs.get("due_date"):
            try:
                from polaris.tools.utils import parse_datetime
                tz = context.config.timezone if context.config else "UTC"
                due_at = parse_datetime(kwargs["due_date"], kwargs.get("due_time"), timezone=tz)
                if due_at.tzinfo:
                    due_at = due_at.replace(tzinfo=None)
                update_kwargs["due_at"] = due_at
            except ValueError as e:
                return f"Error parsing due date: {e}"
        if kwargs.get("tags"):
            update_kwargs["tags"] = [t.strip() for t in kwargs["tags"].split(",") if t.strip()]
        if kwargs.get("project"):
            update_kwargs["project_id"] = kwargs["project"]

        if not update_kwargs:
            return "No fields to update. Specify at least one field to change."

        try:
            todo = await context.todo_manager.update(todo_id, **update_kwargs)
            if not todo:
                return f"Todo not found: {todo_id}"

            return f"Updated todo: **{todo.title}** [{todo.id}]\n{todo.format_for_display()}"

        except Exception as e:
            logger.error(f"Error updating todo: {e}", exc_info=True)
            return f"Error updating todo: {e}"


class CompleteTodoTool(Tool):
    """Tool for marking a todo as completed."""

    @property
    def name(self) -> str:
        return "complete_todo"

    @property
    def description(self) -> str:
        return "Mark a todo as completed. Provide the todo ID."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="todo_id",
                type="string",
                description="The todo ID to complete (e.g., 'todo-1')",
                required=True,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.todo_manager:
            return "Error: Todo system not available. Redis is required."

        todo_id = kwargs.get("todo_id")
        if not todo_id:
            return "Error: todo_id is required."

        try:
            todo = await context.todo_manager.complete(todo_id)
            if not todo:
                return f"Todo not found: {todo_id}"

            return f"Completed: **{todo.title}** [{todo.id}]"

        except Exception as e:
            logger.error(f"Error completing todo: {e}", exc_info=True)
            return f"Error completing todo: {e}"


class DeleteTodoTool(Tool):
    """Tool for deleting a todo."""

    @property
    def name(self) -> str:
        return "delete_todo"

    @property
    def description(self) -> str:
        return "Permanently delete a todo item. This cannot be undone."

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="todo_id",
                type="string",
                description="The todo ID to delete (e.g., 'todo-1')",
                required=True,
            ),
        ]

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.todo_manager:
            return "Error: Todo system not available. Redis is required."

        todo_id = kwargs.get("todo_id")
        if not todo_id:
            return "Error: todo_id is required."

        try:
            # Get the todo first for confirmation message
            todo = await context.todo_manager.get(todo_id)
            if not todo:
                return f"Todo not found: {todo_id}"

            title = todo.title
            success = await context.todo_manager.delete(todo_id)
            if success:
                return f"Deleted todo: **{title}** [{todo_id}]"
            else:
                return f"Failed to delete todo: {todo_id}"

        except Exception as e:
            logger.error(f"Error deleting todo: {e}", exc_info=True)
            return f"Error deleting todo: {e}"


class GetTodoSummaryTool(Tool):
    """Tool for getting a dashboard summary of todos."""

    @property
    def name(self) -> str:
        return "get_todo_summary"

    @property
    def description(self) -> str:
        return (
            "Get a summary/dashboard of the user's todos: counts by status, "
            "overdue items, items due today, and overall stats."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: PolarisToolContext, **kwargs) -> str:
        if not context.todo_manager:
            return "Error: Todo system not available. Redis is required."

        user_id = str(context.user_id) if context.user_id else "default"

        try:
            summary = await context.todo_manager.get_summary(user_id)

            lines = ["**Todo Summary**\n"]
            lines.append(f"Total: {summary['total']} items")

            # Status breakdown
            status_line = []
            for status, count in summary['by_status'].items():
                if count > 0:
                    status_line.append(f"{status}: {count}")
            if status_line:
                lines.append(f"By status: {', '.join(status_line)}")

            # Overdue
            if summary['overdue_count'] > 0:
                lines.append(f"\n**Overdue ({summary['overdue_count']}):**")
                for item_str in summary['overdue']:
                    lines.append(item_str)

            # Due today
            if summary['due_today_count'] > 0:
                lines.append(f"\n**Due Today ({summary['due_today_count']}):**")
                for item_str in summary['due_today']:
                    lines.append(item_str)

            if summary['total'] == 0:
                lines.append("\nNo todos yet. Create one to get started!")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error getting summary: {e}", exc_info=True)
            return f"Error getting todo summary: {e}"
