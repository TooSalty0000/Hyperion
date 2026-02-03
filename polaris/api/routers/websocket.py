"""WebSocket router for real-time todo updates via Redis Pub/Sub."""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from polaris.api.auth import verify_token
from polaris.api.config import get_api_config
from polaris.api.dependencies import get_redis

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/todos")
async def todo_websocket(websocket: WebSocket, token: str = Query(...)):
    """
    WebSocket endpoint for real-time todo updates.

    Subscribes to Redis Pub/Sub channel for the authenticated user
    and forwards events to the client.

    Connect with: ws://host:port/ws/todos?token=<jwt>
    """
    # Validate JWT before accepting connection
    config = get_api_config()
    if not config.secret_key:
        await websocket.close(code=1008, reason="Server not configured")
        return

    payload = verify_token(token, config.secret_key)
    if payload is None:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return

    user_id = payload.get("sub", "owner")
    channel_name = f"hyperion:todo:events:{user_id}"

    await websocket.accept()
    logger.info(f"WebSocket connected for user {user_id}")

    # Create a dedicated Redis subscriber
    import redis.asyncio as redis_async

    subscriber = await redis_async.from_url(
        config.redis_url, encoding="utf-8", decode_responses=True
    )
    pubsub = subscriber.pubsub()
    await pubsub.subscribe(channel_name)

    try:
        # Run two concurrent tasks:
        # 1. Forward Redis Pub/Sub messages to WebSocket
        # 2. Read WebSocket messages (for keepalive pings)
        await asyncio.gather(
            _forward_pubsub(pubsub, websocket),
            _read_websocket(websocket),
        )
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    finally:
        await pubsub.unsubscribe(channel_name)
        await pubsub.aclose()
        await subscriber.aclose()


async def _forward_pubsub(pubsub, websocket: WebSocket):
    """Forward Redis Pub/Sub messages to the WebSocket client."""
    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                await websocket.send_text(message["data"])
            except Exception:
                break


async def _read_websocket(websocket: WebSocket):
    """Read from WebSocket to detect disconnects and handle pings."""
    try:
        while True:
            data = await websocket.receive_text()
            # Client can send "ping" for keepalive
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        raise
    except Exception:
        pass
