import hmac

from fastapi import (
    Depends,
    Header,
    HTTPException
)
from fastapi.security import (
    OAuth2PasswordBearer
)

from config import INTERNAL_API_KEY
from database import get_db
from models import User
from security import InvalidTokenError, decode_access_token


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/users/login"
)
optional_oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/users/login",
    auto_error=False,
)


def credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={
            "WWW-Authenticate": "Bearer"
        }
    )


def get_api_key(
    api_key: str = Header()
):
    # No key configured means the internal API is disabled.
    if not INTERNAL_API_KEY or not hmac.compare_digest(
        api_key.encode(),
        INTERNAL_API_KEY.encode(),
    ):
        raise HTTPException(
            status_code=401,
            detail="invalid api key"
        )

    return api_key


def _user_from_token(
    token: str,
    db,
) -> User:

    try:
        payload = decode_access_token(token)
    except InvalidTokenError:
        raise credentials_exception()

    # The role is read from the database, not trusted from the token,
    # so a role change or deleted user takes effect immediately.
    user = db.get(
        User,
        payload["user_id"]
    )

    if user is None:
        raise credentials_exception()

    return user


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db=Depends(get_db)
):
    return _user_from_token(token, db)


def get_optional_current_user(
    token: str | None = Depends(optional_oauth2_scheme),
    db=Depends(get_db),
):
    if token is None:
        return None

    return _user_from_token(token, db)


def require_admin(
    user=Depends(get_current_user)
):
    if user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="admin permission required"
        )

    return user


def check_permission(
    user=Depends(get_current_user)
):
    return user
