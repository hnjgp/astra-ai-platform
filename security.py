from datetime import datetime, timedelta, timezone

from fastapi import (
    Depends,
    HTTPException,
    status
)

from fastapi.security import OAuth2PasswordBearer

from jose import jwt, JWTError

from pwdlib import PasswordHash

from config import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES
)


password_hash = PasswordHash.recommended()


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/users/login"
)


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
    data: dict
):

    to_encode = data.copy()

    expire = (
        datetime.now(timezone.utc)
        +
        timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    )

    to_encode["exp"] = expire

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )



def decode_access_token(
    token: str = Depends(oauth2_scheme)
):

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[
                ALGORITHM
            ]
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid token",
                headers={
                    "WWW-Authenticate": "Bearer"
                }
            )


        return payload


    except JWTError:

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={
                "WWW-Authenticate": "Bearer"
            }
        )