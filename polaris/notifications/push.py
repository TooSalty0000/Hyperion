"""Web Push notification service using VAPID/pywebpush."""

import json
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

_redis = None


def _import_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as redis_async
            _redis = redis_async
        except ImportError:
            raise ImportError("redis is required for PushNotificationService.")
    return _redis


class PushNotificationService:
    """
    Sends Web Push notifications via pywebpush.

    Manages subscriptions stored in Redis and handles stale subscription cleanup.

    Redis keys:
    - hyperion:push:subscriptions                -> SET of subscription JSON strings
    - hyperion:push:notification_log             -> LIST of recent notifications (capped)
    """

    SUBSCRIPTIONS_KEY = "hyperion:push:subscriptions"
    LOG_KEY = "hyperion:push:notification_log"
    MAX_LOG_ENTRIES = 100

    def __init__(
        self,
        redis_url: str,
        vapid_private_key: str,
        vapid_public_key: str,
        vapid_subject: str,
    ):
        self._redis_url = redis_url
        self._redis = None
        self._vapid_private_key = vapid_private_key
        self._vapid_public_key = vapid_public_key
        self._vapid_subject = vapid_subject

    @property
    def is_configured(self) -> bool:
        """Check if VAPID keys are configured."""
        return bool(self._vapid_private_key and self._vapid_public_key and self._vapid_subject)

    async def _get_client(self):
        if self._redis is None:
            redis_async = _import_redis()
            self._redis = await redis_async.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
        return self._redis

    async def get_subscriptions(self) -> List[Dict[str, Any]]:
        """Get all registered push subscriptions."""
        redis = await self._get_client()
        raw = await redis.smembers(self.SUBSCRIPTIONS_KEY)
        subs = []
        for s in raw:
            try:
                subs.append(json.loads(s))
            except json.JSONDecodeError:
                pass
        return subs

    async def send_notification(
        self,
        title: str,
        body: str,
        tag: Optional[str] = None,
        url: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Send a push notification to all registered subscriptions.

        Returns dict with 'sent', 'failed', 'stale_removed' counts.
        """
        if not self.is_configured:
            logger.warning("Push notifications not configured (missing VAPID keys)")
            return {"sent": 0, "failed": 0, "stale_removed": 0}

        try:
            from pywebpush import webpush, WebPushException
        except ImportError:
            logger.error("pywebpush not installed")
            return {"sent": 0, "failed": 0, "stale_removed": 0}

        subscriptions = await self.get_subscriptions()
        if not subscriptions:
            return {"sent": 0, "failed": 0, "stale_removed": 0}

        payload = json.dumps({
            "title": title,
            "body": body,
            "tag": tag or "polaris",
            "url": url,
            "data": data,
        })

        sent = 0
        failed = 0
        stale = []
        redis = await self._get_client()

        for sub in subscriptions:
            try:
                webpush(
                    subscription_info=sub,
                    data=payload,
                    vapid_private_key=self._vapid_private_key,
                    vapid_claims={"sub": self._vapid_subject},
                )
                sent += 1
            except WebPushException as e:
                logger.warning(f"Push failed: {e}")
                if hasattr(e, 'response') and e.response and e.response.status_code == 410:
                    stale.append(json.dumps(sub))
                failed += 1
            except Exception as e:
                logger.error(f"Push error: {e}")
                failed += 1

        # Clean up stale subscriptions
        for s in stale:
            await redis.srem(self.SUBSCRIPTIONS_KEY, s)

        # Log notification
        log_entry = json.dumps({
            "title": title,
            "body": body,
            "sent": sent,
            "failed": failed,
        })
        await redis.lpush(self.LOG_KEY, log_entry)
        await redis.ltrim(self.LOG_KEY, 0, self.MAX_LOG_ENTRIES - 1)

        return {"sent": sent, "failed": failed, "stale_removed": len(stale)}
