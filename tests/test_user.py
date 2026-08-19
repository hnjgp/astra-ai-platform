def test_register_user(client):

    response = client.post(
        "/users/register",
        json={
            "username": "ali",
            "password": "123456"
        }
    )

    assert response.status_code == 200
    assert response.json()["username"] == "ali"



def test_duplicate_username(client):

    client.post(
        "/users/register",
        json={
            "username": "ali",
            "password": "123456"
        }
    )


    response = client.post(
        "/users/register",
        json={
            "username": "ali",
            "password": "999999"
        }
    )


    assert response.status_code == 400



def test_register_without_password(client):

    response = client.post(
        "/users/register",
        json={
            "username": "reza"
        }
    )


    assert response.status_code == 422



def test_login_user(client):

    client.post(
        "/users/register",
        json={
            "username": "ali",
            "password": "123456"
        }
    )


    response = client.post(
        "/users/login",
        data={
            "username": "ali",
            "password": "123456"
        }
    )


    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"



def test_login_wrong_password(client):

    client.post(
        "/users/register",
        json={
            "username": "ali",
            "password": "123456"
        }
    )


    response = client.post(
        "/users/login",
        data={
            "username": "ali",
            "password": "999999"
        }
    )


    assert response.status_code == 401



def test_login_wrong_username(client):

    response = client.post(
        "/users/login",
        data={
            "username": "unknown",
            "password": "123456"
        }
    )


    assert response.status_code == 401
    assert response.json()["detail"] == "incorrect username or password"



def test_profile_without_token(client):

    response = client.get(
        "/users/profile"
    )


    assert response.status_code == 401



def test_profile_with_token(client):

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
            "username": "ali",
            "password": "123456"
        }
    )


    token = login.json()["access_token"]


    response = client.get(
        "/users/profile",
        headers={
            "Authorization": f"Bearer {token}"
        }
    )


    assert response.status_code == 200
    assert response.json()["username"] == "ali"



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
            "username": "ali",
            "password": "123456"
        }
    )


    token = login.json()["access_token"]


    response = client.post(
        "/users/admin",
        json={
            "username": "admin1",
            "password": "123456"
        },
        headers={
            "Authorization": f"Bearer {token}"
        }
    )


    assert response.status_code == 403