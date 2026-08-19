from services.user_service import create_admin
from database import SessionLocal
from models import User


def create_admin_token(client):

    from database import SessionLocal
    from models import User
    from services.user_service import create_admin


    db = SessionLocal()


    existing = db.query(User).filter(
        User.username == "admin"
    ).first()


    if existing:
        db.delete(existing)
        db.commit()


    create_admin(
        db,
        "admin",
        "123456"
    )


    db.close()


    response = client.post(
        "/users/login",
        data={
            "username": "admin",
            "password": "123456"
        }
    )


    print(response.json())


    assert response.status_code == 200


    return response.json()["access_token"]



def test_admin_can_create_admin(client):

    token = create_admin_token(client)


    response = client.post(
        "/users/admin",
        headers={
            "Authorization": f"Bearer {token}"
        },
        json={
            "username": "admin2",
            "password": "123456"
        }
    )


    assert response.status_code == 200

    assert response.json()["username"] == "admin2"

    assert response.json()["role"] == "admin"




def test_normal_user_cannot_create_admin(client):

    client.post(
        "/users/register",
        json={
            "username": "ali",
            "password": "123456"
        }
    )


    login = client.post(
        "/users/login",
        data={
            "username":"ali",
            "password":"123456"
        }
    )


    token = login.json()["access_token"]


    response = client.post(
        "/users/admin",
        headers={
            "Authorization": f"Bearer {token}"
        },
        json={
            "username":"admin3",
            "password":"123456"
        }
    )


    assert response.status_code == 403




def test_admin_can_delete_document(client):

    token = create_admin_token(client)


    create_response = client.post(
        "/documents/",
        headers={
            "Authorization": f"Bearer {token}"
        },
        json={
            "title":"delete test",
            "category":"test"
        }
    )


    assert create_response.status_code == 200


    document_id = create_response.json()["id"]


    response = client.delete(
        f"/documents/{document_id}",
        headers={
            "Authorization": f"Bearer {token}"
        }
    )


    assert response.status_code == 200

    assert response.json()["message"] == "document deleted"