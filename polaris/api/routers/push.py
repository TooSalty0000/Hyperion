"""Web Push subscription management router."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException

from polaris.api.dependencies import require_auth, get_redis
from polaris.api.config import get_api_config
from polaris.api.schemas import PushSubscription, VapidKeyResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/push", tags=["push"])

SUBSCRIPTIONS_KEY = "hyperion:push:subscriptions"


@router.get("/vapid-key", response_model=VapidKeyResponse)
async def get_vapid_key(payload: dict = Depends(require_auth)):
    """Return the VAPID public key for client-side push subscription."""
    config = get_api_config()
    if not config.vapid_public_key:
        raise HTTPException(
            status_code=503,
            detail="VAPID keys not configured. Run: python -m polaris.api.setup_vapid",
        )
    return VapidKeyResponse(public_key=config.vapid_public_key)


@router.post("/subscribe", status_code=201)
async def subscribe(
    body: PushSubscription,
    payload: dict = Depends(require_auth),
    redis=Depends(get_redis),
):
    """Register a Web Push subscription."""
    sub_json = json.dumps({
        "endpoint": body.endpoint,
        "keys": body.keys,
    })
    await redis.sadd(SUBSCRIPTIONS_KEY, sub_json)
    logger.info(f"Push subscription registered: {body.endpoint[:60]}...")
    return {"status": "subscribed"}


@router.post("/unsubscribe", status_code=200)
async def unsubscribe(
    body: PushSubscription,
    payload: dict = Depends(require_auth),
    redis=Depends(get_redis),
):
    """Remove a Web Push subscription."""
    sub_json = json.dumps({
        "endpoint": body.endpoint,
        "keys": body.keys,
    })
    await redis.srem(SUBSCRIPTIONS_KEY, sub_json)
    return {"status": "unsubscribed"}


@router.post("/test")
async def send_test_notification(
    payload: dict = Depends(require_auth),
    redis=Depends(get_redis),
):
    """Send a test push notification to all registered subscriptions."""
    config = get_api_config()
    if not config.vapid_private_key:
        raise HTTPException(status_code=503, detail="VAPID keys not configured")

    subs_raw = await redis.smembers(SUBSCRIPTIONS_KEY)
    if not subs_raw:
        raise HTTPException(status_code=404, detail="No push subscriptions registered")

    sent = 0
    failed = 0

    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        raise HTTPException(status_code=503, detail="pywebpush not installed")

    notification_data = json.dumps({
        "title": "Polaris Test",
        "body": "Push notifications are working!",
        "tag": "test",
    })

    stale_subs = []
    for sub_json in subs_raw:
        sub = json.loads(sub_json)
        try:
            webpush(
                subscription_info=sub,
                data=notification_data,
                vapid_private_key=config.vapid_private_key,
                vapid_claims={"sub": config.vapid_subject},
            )
            sent += 1
        except WebPushException as e:
            logger.warning(f"Push failed for {sub['endpoint'][:60]}: {e}")
            # Remove stale subscriptions (410 Gone, 404 Not Found)
            if "410" in str(e) or "404" in str(e):
                stale_subs.append(sub_json)
            failed += 1
        except Exception as e:
            logger.error(f"Push error: {e}")
            failed += 1

    # Clean up stale subscriptions
    for stale in stale_subs:
        await redis.srem(SUBSCRIPTIONS_KEY, stale)

    return {"sent": sent, "failed": failed, "stale_removed": len(stale_subs)}
