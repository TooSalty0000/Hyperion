"""JWT authentication for the Polaris API."""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

logger = logging.getLogger(__name__)

# Token expiry
TOKEN_EXPIRY_DAYS = 30


def create_token(secret_key: str, user_id: str = "owner") -> str:
    """Create a JWT token."""
    payload = {
        "sub": user_id,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRY_DAYS),
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")


def verify_token(token: str, secret_key: str) -> Optional[dict]:
    """Verify a JWT token. Returns payload or None."""
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid token: {e}")
        return None


async def verify_password(password: str, redis_client) -> bool:
    """Verify password against stored bcrypt hash."""
    stored_hash = await redis_client.get("polaris:app:password_hash")
    if not stored_hash:
        return False

    try:
        from pwdlib.hashers.bcrypt import BcryptHasher

        hasher = BcryptHasher()
        return hasher.verify(password, stored_hash)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False


async def set_password(password: str, redis_client) -> bool:
    """Hash and store a password."""
    try:
        from pwdlib.hashers.bcrypt import BcryptHasher

        hasher = BcryptHasher()
        hashed = hasher.hash(password)
        await redis_client.set("polaris:app:password_hash", hashed)
        return True
    except Exception as e:
        logger.error(f"Password hashing failed: {e}")
        return False


async def has_password(redis_client) -> bool:
    """Check if a password has been set."""
    return await redis_client.exists("polaris:app:password_hash") > 0
