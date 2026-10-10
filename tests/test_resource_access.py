def token_headers(client, username):
    client.post(
        "/users/register",
        json={"username": username, "password": "secret123"},
    )

    token = client.post(
        "/users/login",
        data={"username": username, "password": "secret123"},
    ).json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


def create_private_document(client, headers):
    return client.post(
        "/documents/",
        json={
            "title": "salary data",
            "category": "HR",
            "body": "confidential",
            "is_private": True,
        },
        headers=headers,
    ).json()["id"]


def test_secure_endpoint_hides_other_users_private_documents(client):
    owner = token_headers(client, "secure_owner")
    other = token_headers(client, "secure_other")

    document_id = create_private_document(client, owner)

    response = client.get(
        f"/documents/secure/{document_id}",
        headers=other,
    )

    assert response.status_code == 403


def test_secure_endpoint_allows_owner(client):
    owner = token_headers(client, "secure_owner2")

    document_id = create_private_document(client, owner)

    response = client.get(
        f"/documents/secure/{document_id}",
        headers=owner,
    )

    assert response.status_code == 200
    assert response.json()["body"] == "confidential"


def test_document_list_hides_other_users_private_documents(client):
    owner = token_headers(client, "list_owner")
    other = token_headers(client, "list_other")

    document_id = create_private_document(client, owner)

    ids = [
        document["id"]
        for document in client.get("/documents/", headers=other).json()
    ]

    assert document_id not in ids


def test_readable_document_ids_follow_ownership(client):
    from database import SessionLocal
    from models import User
    from repository.document import get_readable_document_ids

    owner = token_headers(client, "ids_owner")
    other = token_headers(client, "ids_other")

    private_id = create_private_document(client, owner)

    public_id = client.post(
        "/documents/",
        json={"title": "public", "category": "AI"},
        headers=other,
    ).json()["id"]

    db = SessionLocal()

    owner_user = db.query(User).filter_by(username="ids_owner").one()
    other_user = db.query(User).filter_by(username="ids_other").one()

    assert set(get_readable_document_ids(db, owner_user)) == {
        private_id,
        public_id,
    }
    assert get_readable_document_ids(db, other_user) == [public_id]

    other_user.role = "admin"
    assert get_readable_document_ids(db, other_user) is None

    db.close()
