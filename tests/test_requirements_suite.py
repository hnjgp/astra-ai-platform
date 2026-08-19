"""Acceptance tests for the public API requirements.

They deliberately exercise HTTP behavior rather than service internals, so the
suite remains useful when the implementation changes.
"""
from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt
from sqlalchemy.exc import IntegrityError

from config import ALGORITHM, SECRET_KEY
from database import SessionLocal
from models import Document, User


def register(client, username="userone", password="secret1"):
    return client.post("/users/register", json={"username": username, "password": password})


def login(client, username="userone", password="secret1"):
    return client.post("/users/login", data={"username": username, "password": password})


def auth(client, username="userone", password="secret1"):
    token = login(client, username, password).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def make_admin(client, username="adminone"):
    db = SessionLocal()
    user = User(username=username, password="placeholder", role="admin")
    # Use the normal registration path for password hashing, then elevate in
    # the isolated test database.
    db.close()
    register(client, username, "secret1")
    db = SessionLocal()
    user = db.query(User).filter_by(username=username).one()
    user.role = "admin"
    db.commit()
    db.close()
    return auth(client, username, "secret1")


def create_document(client, headers=None, **extra):
    payload = {"title": "document", "category": "AI", "body": "body"}
    payload.update(extra)
    return client.post("/documents/", json=payload, headers=headers or {})


def test_register_success(client):
    assert register(client).status_code == 200


@pytest.mark.parametrize("payload", [
    {"username": "", "password": "secret1"},
    {"username": "bad name", "password": "secret1"},
    {"username": "valid", "password": "123"},
])
def test_register_validation(client, payload):
    assert client.post("/users/register", json=payload).status_code == 422


def test_register_duplicate_username(client):
    register(client)
    assert register(client).status_code == 400


def test_password_is_hashed(client):
    register(client)
    db = SessionLocal()
    saved = db.query(User).filter_by(username="userone").one()
    assert saved.password != "secret1" and saved.password.startswith("$argon2")
    db.close()


def test_login_and_token_claims(client):
    register(client)
    response = login(client)
    assert response.status_code == 200
    token = response.json()["access_token"]
    claims = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert response.json()["token_type"] == "bearer"
    assert claims["sub"] and claims["role"] == "user"


@pytest.mark.parametrize("username,password", [("userone", "wrongpw"), ("unknown", "secret1")])
def test_login_wrong_credentials(client, username, password):
    register(client)
    assert login(client, username, password).status_code == 401


def test_login_empty_credentials(client):
    assert client.post("/users/login", data={}).status_code == 422


def test_expired_invalid_and_tampered_token(client):
    expired = jwt.encode({"sub": "1", "role": "user", "exp": datetime.now(timezone.utc) - timedelta(minutes=1)}, SECRET_KEY, algorithm=ALGORITHM)
    for token in (expired, "invalid.jwt.token", expired + "x"):
        assert client.get("/users/profile", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_profile_authentication(client):
    register(client)
    assert client.get("/users/profile").status_code == 401
    response = client.get("/users/profile", headers=auth(client))
    assert response.status_code == 200 and response.json()["username"] == "userone"


def test_admin_management(client):
    headers = make_admin(client)
    created = client.post("/users", headers=headers, json={"username": "member", "password": "secret1"})
    assert created.status_code == 200
    user_id = created.json()["id"]
    assert client.post("/users/admin", headers=headers, json={"username": "secondadmin", "password": "secret1"}).json()["role"] == "admin"
    assert client.post("/users", headers=headers, json={"username": "member", "password": "secret1"}).status_code == 400
    assert client.get("/users", headers=headers).status_code == 200
    assert client.get(f"/users/{user_id}", headers=headers).status_code == 200
    assert client.patch(f"/users/{user_id}/role", headers=headers, json={}).status_code == 422
    assert client.patch(f"/users/{user_id}/role", headers=headers, json={"role": "admin"}).status_code == 200
    assert client.delete(f"/users/{user_id}", headers=headers).status_code == 200


def test_normal_user_admin_protection(client):
    register(client)
    headers = auth(client)
    assert client.post("/users/admin", headers=headers, json={"username": "noadmin", "password": "secret1"}).status_code == 403
    assert client.get("/users", headers=headers).status_code == 403


def test_document_crud_and_validation(client):
    for payload in ({"category": "AI"}, {"title": "x"}, {"title": "x", "category": "AI", "body": ""}, {"title": "x", "category": "AI", "body": 1}):
        assert client.post("/documents/", json=payload).status_code == 422
    response = create_document(client)
    document = response.json()
    assert response.status_code == 200 and document["body"] == "body"
    assert len(client.get("/documents/").json()) == 1
    assert client.get(f"/documents/{document['id']}").status_code == 200
    assert client.get("/documents/9999").status_code == 404
    updated = client.put(f"/documents/{document['id']}", json={"title": "changed"})
    assert updated.status_code == 200 and updated.json()["title"] == "changed"
    assert client.put("/documents/9999", json={"title": "changed"}).status_code == 404


def test_document_permissions_and_admin(client):
    register(client, "owner", "secret1")
    register(client, "other", "secret1")
    owner = auth(client, "owner", "secret1")
    other = auth(client, "other", "secret1")
    document = create_document(client, owner, title="private", is_private=True).json()
    doc_id = document["id"]
    assert client.get(f"/documents/{doc_id}").status_code == 401
    assert client.get(f"/documents/{doc_id}", headers=other).status_code == 403
    assert client.put(f"/documents/{doc_id}", headers=other, json={"title": "no"}).status_code == 403
    assert client.delete(f"/documents/{doc_id}", headers=other).status_code == 403
    assert client.put(f"/documents/{doc_id}", headers=owner, json={"title": "yes"}).status_code == 200
    headers = make_admin(client)
    assert client.get(f"/documents/{doc_id}", headers=headers).status_code == 200
    assert client.delete(f"/documents/{doc_id}", headers=headers).status_code == 200
    assert client.delete(f"/documents/{doc_id}", headers=headers).status_code == 404


def test_api_validation_security_and_health(client):
    assert client.post("/users/register", content="{").status_code == 422
    assert client.post("/users/register", json={"username": "safe", "password": "secret1", "extra": True}).status_code == 422
    assert register(client, "sql' OR 1=1--", "secret1").status_code == 422
    assert create_document(client, title="<script>alert(1)</script>").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}
    cors = client.options("/health", headers={"Origin": "http://localhost", "Access-Control-Request-Method": "GET"})
    assert cors.headers["access-control-allow-origin"] == "http://localhost"


def test_brute_force_and_rate_limit(client):
    register(client)
    for _ in range(5):
        assert login(client, "userone", "wrongpw").status_code == 401
    assert login(client, "userone", "wrongpw").status_code == 429
    for _ in range(120):
        assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 429


def test_database_transaction_rollback_and_tables(client):
    db = SessionLocal()
    assert User.__tablename__ in User.metadata.tables and Document.__tablename__ in Document.metadata.tables
    db.add(User(username="transaction", password="x", role="user"))
    db.rollback()
    assert db.query(User).filter_by(username="transaction").first() is None
    db.close()


def test_database_foreign_key_constraints(client):
    db = SessionLocal()
    db.add(Document(title="invalid", category="AI", owner_id=99999))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    db.close()


def test_multiple_users_registration(client):
    for index in range(20):
        assert register(client, f"user{index:02d}", "secret1").status_code == 200


def test_multiple_documents_creation_and_large_payload(client):
    body = "x" * 100_000
    assert create_document(client, title="large", body=body).status_code == 200
    for index in range(10):
        assert create_document(client, title=f"doc-{index}").status_code == 200
    assert len(client.get("/documents/").json()) == 11


def test_api_response_time(client):
    from time import monotonic
    start = monotonic()
    assert client.get("/health").status_code == 200
    assert monotonic() - start < 1


def test_environment_configuration_loaded():
    from config import DATABASE_URL
    assert DATABASE_URL


def test_missing_secret_key(monkeypatch):
    from pydantic import ValidationError
    from config import Settings
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, DATABASE_URL="sqlite:///test.db")


def test_database_url_configuration():
    from config import Settings
    assert Settings(_env_file=None, SECRET_KEY="test", DATABASE_URL="sqlite:///configured.db").DATABASE_URL.endswith("configured.db")


def test_error_response_format_and_sensitive_data(client):
    response = register(client)
    assert "password" not in response.json()
    error = register(client)
    assert error.status_code == 400 and set(error.json()) == {"detail"}


def test_public_document_access(client):
    document = create_document(client).json()
    assert client.get(f"/documents/{document['id']}").status_code == 200


def test_extra_fields_and_wrong_field_types(client):
    assert client.post("/documents/", json={"title": "x", "category": "AI", "unknown": 1}).status_code == 422
    assert client.post("/documents/", json={"title": ["not", "text"], "category": "AI"}).status_code == 422


def test_database_cleanup_between_tests(client):
    # The autouse fixture creates a fresh schema.  This test starts empty even
    # though neighboring tests persist users/documents within their own test.
    assert client.get("/documents/").json() == []


def test_docker_artifacts_exist():
    from pathlib import Path
    root = Path(__file__).parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "uvicorn" in dockerfile and (root / ".dockerignore").is_file()
