"""FastAPI dependency injection for shared resources."""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Header, status

from polaris.api.config import get_api_config, APIConfig
from polaris.api.auth import verify_token
from polaris.todo.manager import TodoManager
from polaris.todo.reminders import ReminderManager

logger = logging.getLogger(__name__)

# Singleton managers (initialized at app startup)
_todo_manager: Optional[TodoManager] = None
_reminder_manager: Optional[ReminderManager] = None
_redis_client = None


async def init_managers(redis_url: str):
    """Initialize manager singletons. Called during app lifespan."""
    global _todo_manager, _reminder_manager, _redis_client

    import redis.asyncio as redis_async
    _redis_client = await redis_async.from_url(redis_url, encoding="utf-8", decode_responses=True)

    _todo_manager = TodoManager(redis_url)
    _reminder_manager = ReminderManager(redis_url)
    logger.info("API managers initialized")


async def shutdown_managers():
    """Clean up managers. Called during app shutdown."""
    global _redis_client
    if _redis_client:
        await _redis_client.aclose()


def get_todo_manager() -> TodoManager:
    if _todo_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _todo_manager


def get_reminder_manager() -> ReminderManager:
    if _reminder_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _reminder_manager


def get_redis():
    if _redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not initialized")
    return _redis_client


async def require_auth(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI dependency that validates JWT from Authorization header.

    Returns the token payload (contains 'sub' = user_id).
    """
    config = get_api_config()

    if not config.secret_key:
        raise HTTPException(
            status_code=500,
            detail="Server not configured: POLARIS_APP_SECRET is not set",
        )

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    payload = verify_token(token, config.secret_key)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload
