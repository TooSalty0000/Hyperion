"""FastAPI server configuration via environment variables."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """Configuration for the Polaris API server."""

    host: str = "0.0.0.0"
    port: int = 6360
    secret_key: str = ""  # JWT signing key
    redis_url: str = "redis://localhost:6379"
    timezone: str = "UTC"

    # VAPID keys for Web Push
    vapid_private_key: str = ""
    vapid_public_key: str = ""
    vapid_subject: str = ""

    # Static files (Next.js export)
    static_dir: Optional[str] = None

    # CORS (dev only)
    cors_origins: list = None

    @classmethod
    def from_env(cls) -> "APIConfig":
        cors_raw = os.getenv("POLARIS_CORS_ORIGINS", "")
        cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()] if cors_raw else []

        # Only serve static files when explicitly configured
        static_dir = os.getenv("POLARIS_STATIC_DIR", "")

        return cls(
            host=os.getenv("POLARIS_APP_HOST", "0.0.0.0"),
            port=int(os.getenv("POLARIS_APP_PORT", "6360")),
            secret_key=os.getenv("POLARIS_APP_SECRET", ""),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            timezone=os.getenv("POLARIS_TIMEZONE", "UTC"),
            vapid_private_key=os.getenv("VAPID_PRIVATE_KEY", ""),
            vapid_public_key=os.getenv("VAPID_PUBLIC_KEY", ""),
            vapid_subject=os.getenv("VAPID_SUBJECT", ""),
            static_dir=static_dir,
            cors_origins=cors_origins,
        )


_config: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    global _config
    if _config is None:
        _config = APIConfig.from_env()
    return _config
