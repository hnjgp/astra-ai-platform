from datetime import datetime, timedelta, timezone
from uuid import uuid4

from jose import JWTError, jwt
from pwdlib import PasswordHash

from config import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES
)


password_hash = PasswordHash.recommended()

ACCESS_TOKEN_TYPE = "access"


class InvalidTokenError(Exception):
    pass


def hash_password(
    password: str
):
    return password_hash.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str
):
    return password_hash.verify(
        plain_password,
        hashed_password
    )


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:

    now = datetime.now(timezone.utc)

    to_encode = data.copy()

    to_encode.update(
        {
            "iat": now,
            "exp": now + (
                expires_delta
                or timedelta(
                    minutes=ACCESS_TOKEN_EXPIRE_MINUTES
                )
            ),
            "jti": uuid4().hex,
            "type": ACCESS_TOKEN_TYPE,
        }
    )

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def decode_access_token(
    token: str,
) -> dict:
    """
    Verify signature, expiry and required claims of an access token.
    """

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={
                "require_exp": True,
                "require_sub": True,
            },
        )

    except JWTError as exc:
        raise InvalidTokenError(
            "invalid token"
        ) from exc

    if payload.get("type") != ACCESS_TOKEN_TYPE:
        raise InvalidTokenError(
            "invalid token type"
        )

    try:
        payload["user_id"] = int(payload["sub"])
    except (TypeError, ValueError) as exc:
        raise InvalidTokenError(
            "invalid token subject"
        ) from exc

    return payload
