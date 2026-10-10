from datetime import timedelta

import pytest
from jose import jwt
from pydantic import ValidationError

from config import ALGORITHM, SECRET_KEY, Settings
from security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
)


def register_and_token(client, username="auth_user"):
    client.post(
        "/users/register",
        json={"username": username, "password": "secret123"},
    )

    return client.post(
        "/users/login",
        data={"username": username, "password": "secret123"},
    ).json()["access_token"]


def profile(client, token):
    return client.get(
        "/users/profile",
        headers={"Authorization": f"Bearer {token}"},
    )


def test_token_has_expiry_issue_time_id_and_type(client):
    payload = jwt.decode(
        register_and_token(client),
        SECRET_KEY,
        algorithms=[ALGORITHM],
    )

    assert {"exp", "iat", "jti", "sub"} <= payload.keys()
    assert payload["type"] == "access"


def test_each_token_has_unique_id():
    first = decode_access_token(create_access_token({"sub": "1"}))
    second = decode_access_token(create_access_token({"sub": "1"}))

    assert first["jti"] != second["jti"]


def test_expired_token_is_rejected(client):
    register_and_token(client)

    token = create_access_token(
        {"sub": "1"},
        expires_delta=timedelta(seconds=-1),
    )

    assert profile(client, token).status_code == 401


def test_token_without_expiry_is_rejected():
    token = jwt.encode(
        {"sub": "1", "type": "access"},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    with pytest.raises(InvalidTokenError):
        decode_access_token(token)


def test_token_of_wrong_type_is_rejected(client):
    register_and_token(client)

    token = create_access_token({"sub": "1", "type": "refresh"})

    # create_access_token always sets type=access; forge a refresh token.
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    payload["type"] = "refresh"
    forged = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    assert profile(client, forged).status_code == 401


def test_token_signed_with_other_key_is_rejected(client):
    register_and_token(client)

    payload = jwt.decode(
        create_access_token({"sub": "1"}),
        SECRET_KEY,
        algorithms=[ALGORITHM],
    )
    forged = jwt.encode(payload, "x" * 40, algorithm=ALGORITHM)

    assert profile(client, forged).status_code == 401


def test_non_numeric_subject_is_rejected():
    with pytest.raises(InvalidTokenError):
        decode_access_token(create_access_token({"sub": "abc"}))


def test_token_of_deleted_user_is_rejected(client):
    token = register_and_token(client, "short_lived")

    from database import SessionLocal
    from models import User

    db = SessionLocal()
    db.query(User).filter(User.username == "short_lived").delete()
    db.commit()
    db.close()

    assert profile(client, token).status_code == 401


def test_role_comes_from_database_not_token(client):
    token = register_and_token(client, "promoted_later")

    from database import SessionLocal
    from models import User

    db = SessionLocal()
    user = db.query(User).filter(User.username == "promoted_later").one()
    user.role = "admin"
    db.commit()
    db.close()

    # The token still says role=user, but the database decides.
    assert profile(client, token).json()["role"] == "admin"


def test_auth_test_does_not_echo_api_key(client, monkeypatch):
    import dependency

    monkeypatch.setattr(dependency, "INTERNAL_API_KEY", "k" * 40)

    response = client.get(
        "/auth-test",
        headers={"api-key": "k" * 40},
    )

    assert response.status_code == 200
    assert response.json() == {"authenticated": True}


def test_wrong_api_key_is_rejected(client, monkeypatch):
    import dependency

    monkeypatch.setattr(dependency, "INTERNAL_API_KEY", "k" * 40)

    response = client.get(
        "/auth-test",
        headers={"api-key": "wrong"},
    )

    assert response.status_code == 401


def test_missing_internal_key_disables_api_key_auth(client, monkeypatch):
    import dependency

    monkeypatch.setattr(dependency, "INTERNAL_API_KEY", None)

    response = client.get(
        "/auth-test",
        headers={"api-key": ""},
    )

    assert response.status_code == 401


def test_short_secret_key_is_refused():
    with pytest.raises(ValidationError, match="at least 32"):
        Settings(
            SECRET_KEY="short",
            DATABASE_URL="sqlite://",
            RAG_DATABASE_URL="postgresql://x",
        )
