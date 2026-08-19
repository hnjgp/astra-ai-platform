from jose import jwt

from config import ALGORITHM, SECRET_KEY
from models import User
from security import create_access_token
from services.user_service import create_admin


def admin_headers(client):
    from database import SessionLocal

    db = SessionLocal()
    create_admin(db, "rootadmin", "secret1")
    db.close()
    login = client.post("/users/login", data={"username": "rootadmin", "password": "secret1"})
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_token_contains_user_id_and_role(client):
    client.post("/users/register", json={"username": "nima", "password": "secret1"})
    token = client.post("/users/login", data={"username": "nima", "password": "secret1"}).json()["access_token"]
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"]
    assert payload["role"] == "user"


def test_invalid_and_tampered_tokens_are_rejected(client):
    for token in ("not-a-jwt", create_access_token({"sub": "99999", "role": "user"}) + "x"):
        response = client.get("/users/profile", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401


def test_admin_user_management(client):
    headers = admin_headers(client)
    created = client.post("/users", headers=headers, json={"username": "newuser", "password": "secret1"})
    assert created.status_code == 200
    user_id = created.json()["id"]
    assert client.get("/users", headers=headers).status_code == 200
    assert client.patch(f"/users/{user_id}/role", headers=headers, json={"role": "admin"}).json()["role"] == "admin"
    assert client.delete(f"/users/{user_id}", headers=headers).status_code == 200


def test_admin_cannot_delete_self(client):
    headers = admin_headers(client)
    profile = client.get("/users/profile", headers=headers).json()
    assert client.delete(f"/users/{profile['id']}", headers=headers).status_code == 400


def test_owned_private_document_permissions(client):
    client.post("/users/register", json={"username": "owner", "password": "secret1"})
    client.post("/users/register", json={"username": "other", "password": "secret1"})
    owner_token = client.post("/users/login", data={"username": "owner", "password": "secret1"}).json()["access_token"]
    other_token = client.post("/users/login", data={"username": "other", "password": "secret1"}).json()["access_token"]
    owner_headers = {"Authorization": f"Bearer {owner_token}"}
    document = client.post("/documents/", headers=owner_headers, json={"title": "private", "category": "AI", "body": "content", "is_private": True}).json()
    assert client.get(f"/documents/{document['id']}").status_code == 401
    assert client.put(f"/documents/{document['id']}", headers={"Authorization": f"Bearer {other_token}"}, json={"title": "nope"}).status_code == 403
    assert client.put(f"/documents/{document['id']}", headers=owner_headers, json={"title": "updated"}).status_code == 200
