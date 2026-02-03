"""Reminder CRUD router."""

from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query

from polaris.api.dependencies import require_auth, get_reminder_manager
from polaris.api.schemas import ReminderCreate, ReminderResponse
from polaris.todo.reminders import ReminderManager

router = APIRouter(prefix="/api/reminders", tags=["reminders"])


def _user_id(payload: dict) -> str:
    return payload.get("sub", "owner")


@router.get("", response_model=List[ReminderResponse])
async def list_reminders(
    limit: int = Query(50, ge=1, le=100),
    payload: dict = Depends(require_auth),
    manager: ReminderManager = Depends(get_reminder_manager),
):
    """List all reminders for the current user (active + recently fired)."""
    user_id = _user_id(payload)
    reminders = await manager.list_all(user_id=user_id, limit=limit)
    return [ReminderResponse.from_reminder(r) for r in reminders]


@router.post("", response_model=ReminderResponse, status_code=201)
async def create_reminder(
    body: ReminderCreate,
    payload: dict = Depends(require_auth),
    manager: ReminderManager = Depends(get_reminder_manager),
):
    """Create a new reminder."""
    user_id = _user_id(payload)

    reminder = await manager.create(
        user_id=user_id,
        message=body.message,
        fire_at=body.fire_at,
        todo_id=body.todo_id,
    )
    return ReminderResponse.from_reminder(reminder)


@router.delete("/{reminder_id}", status_code=204)
async def dismiss_reminder(
    reminder_id: str,
    payload: dict = Depends(require_auth),
    manager: ReminderManager = Depends(get_reminder_manager),
):
    """Dismiss (cancel) a reminder."""
    success = await manager.dismiss(reminder_id)
    if not success:
        raise HTTPException(status_code=404, detail="Reminder not found")


@router.post("/{reminder_id}/snooze", response_model=ReminderResponse)
async def snooze_reminder(
    reminder_id: str,
    minutes: int = Query(15, ge=1, le=1440),
    payload: dict = Depends(require_auth),
    manager: ReminderManager = Depends(get_reminder_manager),
):
    """Snooze a reminder by N minutes."""
    reminder = await manager.snooze(reminder_id, minutes=minutes)
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return ReminderResponse.from_reminder(reminder)
