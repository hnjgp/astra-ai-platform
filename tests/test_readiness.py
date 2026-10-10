import main


def test_health_is_liveness_only(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_when_databases_respond(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "database": "ok",
            "rag_database": "ok",
        },
    }


def test_not_ready_when_rag_database_is_down(client, monkeypatch):
    def broken():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(main, "check_rag_database", broken)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {
        "database": "ok",
        "rag_database": "unavailable",
    }
    # Internal error details are logged, not returned.
    assert "refused" not in response.text
