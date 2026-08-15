from datetime import datetime, timedelta, timezone
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, API_KEY
from fastapi import (
    Depends,
    Header,
    HTTPException
)

from fastapi.security import (
    OAuth2PasswordBearer
)

from jose import JWTError, jwt

from database import SessionLocal

from models import User

from config import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES
)


def get_api_key(
    api_key: str = Header()
):
    if api_key == API_KEY:
        return api_key

    raise HTTPException(
        status_code=401,
        detail="unauthorized"
    )


def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


def create_access_token(
    data: dict
):
    to_encode = data.copy()

    expire = (
        datetime.now(timezone.utc)
        + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    )

    to_encode["exp"] = expire

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/users/login"
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db=Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={
            "WWW-Authenticate": "Bearer"
        }
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        user = db.get(
            User,
            int(user_id)
        )

        if user is None:
            raise credentials_exception

        return user

    except (JWTError, ValueError):
        raise credentials_exception


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