"""Todo CRUD router."""

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query

from polaris.api.dependencies import require_auth, get_todo_manager
from polaris.api.schemas import (
    TodoCreate, TodoUpdate, TodoResponse, TodoSummaryResponse,
    PRIORITY_MAP, TodoStatusEnum, TodoPriorityEnum,
)
from polaris.todo.models import TodoStatus, TodoPriority
from polaris.todo.manager import TodoManager

router = APIRouter(prefix="/api/todos", tags=["todos"])


def _user_id(payload: dict) -> str:
    return payload.get("sub", "owner")


@router.get("", response_model=List[TodoResponse])
async def list_todos(
    status: Optional[TodoStatusEnum] = None,
    priority: Optional[TodoPriorityEnum] = None,
    project: Optional[str] = None,
    tag: Optional[str] = None,
    sort_by: str = Query("priority", pattern="^(priority|due|created)$"),
    limit: int = Query(50, ge=1, le=200),
    payload: dict = Depends(require_auth),
    manager: TodoManager = Depends(get_todo_manager),
):
    """List todos with optional filters."""
    user_id = _user_id(payload)

    status_filter = TodoStatus(status.value) if status else None
    priority_filter = TodoPriority(PRIORITY_MAP[priority.value]) if priority else None

    todos = await manager.list_todos(
        user_id=user_id,
        status=status_filter,
        priority=priority_filter,
        project_id=project,
        tag=tag,
        sort_by=sort_by,
        limit=limit,
    )
    return [TodoResponse.from_todo(t) for t in todos]


@router.post("", response_model=TodoResponse, status_code=201)
async def create_todo(
    body: TodoCreate,
    payload: dict = Depends(require_auth),
    manager: TodoManager = Depends(get_todo_manager),
):
    """Create a new todo."""
    user_id = _user_id(payload)
    priority = TodoPriority(PRIORITY_MAP[body.priority.value])

    todo = await manager.create(
        user_id=user_id,
        title=body.title,
        description=body.description,
        priority=priority,
        tags=body.tags,
        project_id=body.project_id,
        parent_id=body.parent_id,
        due_at=body.due_at,
        scheduled_at=body.scheduled_at,
        estimated_duration_minutes=body.estimated_duration_minutes,
    )
    return TodoResponse.from_todo(todo)


@router.get("/summary", response_model=TodoSummaryResponse)
async def get_summary(
    payload: dict = Depends(require_auth),
    manager: TodoManager = Depends(get_todo_manager),
):
    """Get todo dashboard summary."""
    user_id = _user_id(payload)
    summary = await manager.get_summary(user_id)
    return TodoSummaryResponse(
        total=summary["total"],
        by_status=summary["by_status"],
        overdue_count=summary["overdue_count"],
        due_today_count=summary["due_today_count"],
    )


@router.get("/{todo_id}", response_model=TodoResponse)
async def get_todo(
    todo_id: str,
    payload: dict = Depends(require_auth),
    manager: TodoManager = Depends(get_todo_manager),
):
    """Get a single todo."""
    todo = await manager.get(todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return TodoResponse.from_todo(todo)


@router.patch("/{todo_id}", response_model=TodoResponse)
async def update_todo(
    todo_id: str,
    body: TodoUpdate,
    payload: dict = Depends(require_auth),
    manager: TodoManager = Depends(get_todo_manager),
):
    """Update a todo."""
    update_kwargs = {}

    if body.title is not None:
        update_kwargs["title"] = body.title
    if body.description is not None:
        update_kwargs["description"] = body.description
    if body.status is not None:
        update_kwargs["status"] = TodoStatus(body.status.value)
    if body.priority is not None:
        update_kwargs["priority"] = TodoPriority(PRIORITY_MAP[body.priority.value])
    if body.tags is not None:
        update_kwargs["tags"] = body.tags
    if body.project_id is not None:
        update_kwargs["project_id"] = body.project_id
    if body.due_at is not None:
        update_kwargs["due_at"] = body.due_at
    if body.scheduled_at is not None:
        update_kwargs["scheduled_at"] = body.scheduled_at
    if body.estimated_duration_minutes is not None:
        update_kwargs["estimated_duration_minutes"] = body.estimated_duration_minutes
    if body.clear_due:
        update_kwargs["clear_due"] = True
    if body.clear_scheduled:
        update_kwargs["clear_scheduled"] = True

    todo = await manager.update(todo_id, **update_kwargs)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return TodoResponse.from_todo(todo)


@router.delete("/{todo_id}", status_code=204)
async def delete_todo(
    todo_id: str,
    payload: dict = Depends(require_auth),
    manager: TodoManager = Depends(get_todo_manager),
):
    """Delete a todo."""
    success = await manager.delete(todo_id)
    if not success:
        raise HTTPException(status_code=404, detail="Todo not found")


@router.post("/{todo_id}/complete", response_model=TodoResponse)
async def complete_todo(
    todo_id: str,
    payload: dict = Depends(require_auth),
    manager: TodoManager = Depends(get_todo_manager),
):
    """Mark a todo as completed."""
    todo = await manager.complete(todo_id)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return TodoResponse.from_todo(todo)
