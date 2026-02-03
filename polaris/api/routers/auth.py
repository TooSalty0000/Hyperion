"""Auth router - login and token verification."""

from fastapi import APIRouter, Depends, HTTPException, status

from polaris.api.config import get_api_config
from polaris.api.auth import create_token, verify_password, has_password
from polaris.api.dependencies import require_auth, get_redis
from polaris.api.schemas import LoginRequest, TokenResponse, AuthStatus

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, redis=Depends(get_redis)):
    """Authenticate with password and get a JWT token."""
    config = get_api_config()

    if not config.secret_key:
        raise HTTPException(status_code=500, detail="POLARIS_APP_SECRET not configured")

    if not await has_password(redis):
        raise HTTPException(
            status_code=400,
            detail="No password set. Run: python -m polaris.api.setup_password",
        )

    if not await verify_password(body.password, redis):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )

    token = create_token(config.secret_key)
    return TokenResponse(token=token)


@router.get("/verify", response_model=AuthStatus)
async def verify(payload: dict = Depends(require_auth)):
    """Verify that the current token is valid."""
    return AuthStatus(authenticated=True, user_id=payload.get("sub", "owner"))
