def test_create_document(client):

    response = client.post(
        "/documents/",
        json={
            "title": "test document",
            "category": "AI"
        }
    )

    assert response.status_code == 200

    assert response.json()["title"] == "test document"

    assert response.json()["category"] == "AI"



def test_get_documents(client):

    client.post(
        "/documents/",
        json={
            "title": "doc1",
            "category": "AI"
        }
    )


    response = client.get(
        "/documents/"
    )


    assert response.status_code == 200

    assert len(response.json()) == 1



def test_get_document(client):

    create_response = client.post(
        "/documents/",
        json={
            "title": "my document",
            "category": "Data"
        }
    )


    document_id = create_response.json()["id"]


    response = client.get(
        f"/documents/{document_id}"
    )


    assert response.status_code == 200

    assert response.json()["id"] == document_id



def test_get_document_not_found(client):

    response = client.get(
        "/documents/999"
    )


    assert response.status_code == 404



def test_create_document_without_title(client):

    response = client.post(
        "/documents/",
        json={
            "category": "AI"
        }
    )


    assert response.status_code == 422



def test_create_document_without_category(client):

    response = client.post(
        "/documents/",
        json={
            "title": "test"
        }
    )


    assert response.status_code == 422



def test_update_document(client):

    create_response = client.post(
        "/documents/",
        json={
            "title": "old title",
            "category": "AI"
        }
    )


    document_id = create_response.json()["id"]


    response = client.put(
        f"/documents/{document_id}",
        json={
            "title": "new title",
            "category": "ML"
        }
    )


    assert response.status_code == 200

    assert response.json()["title"] == "new title"

    assert response.json()["category"] == "ML"



def test_delete_document_without_admin(client):

    create_response = client.post(
        "/documents/",
        json={
            "title": "delete test",
            "category": "AI"
        }
    )


    document_id = create_response.json()["id"]


    response = client.delete(
        f"/documents/{document_id}"
    )


    assert response.status_code == 401



def test_secure_document_without_permission(client):

    create_response = client.post(
        "/documents/",
        json={
            "title": "secure test",
            "category": "AI"
        }
    )


    document_id = create_response.json()["id"]


    response = client.get(
        f"/documents/secure/{document_id}"
    )


    assert response.status_code == 401