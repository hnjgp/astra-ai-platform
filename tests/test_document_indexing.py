import pytest

from main import app
from rag.chunking.chunker import DocumentChunker
from rag.database import RAGSessionLocal
from rag.indexer import DocumentIndexer
from rag.models import DocumentChunk
from rag.retrieval.retriever import VectorRetriever
from services.providers import get_document_indexer
from tests.fakes import FakeEmbedder


@pytest.fixture
def indexer(rag_db_session):
    indexer = DocumentIndexer(
        embedder=FakeEmbedder(),
        session_factory=RAGSessionLocal,
        chunker=DocumentChunker(
            chunk_size=60,
            chunk_overlap=10,
        ),
    )

    app.dependency_overrides[get_document_indexer] = (
        lambda: indexer
    )

    yield indexer

    app.dependency_overrides.pop(get_document_indexer, None)


def login(client, username):
    client.post(
        "/users/register",
        json={"username": username, "password": "secret123"},
    )

    token = client.post(
        "/users/login",
        data={"username": username, "password": "secret123"},
    ).json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


def chunk_count(session, document_id):
    return (
        session.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .count()
    )


TEXT = (
    "Astra stores document embeddings in PostgreSQL with pgvector. "
    "The agent can search the knowledge base when it needs internal facts."
)


def test_indexer_stores_searchable_chunks(indexer, rag_db_session):
    count = indexer.index(document_id=501, text=TEXT)

    assert count == chunk_count(rag_db_session, 501)
    assert count > 1

    results = VectorRetriever(rag_db_session).search(
        query_embedding=FakeEmbedder().embed("pgvector embeddings"),
        top_k=1,
    )

    assert results[0]["document_id"] == 501
    assert "pgvector" in results[0]["content"]


def test_reindexing_replaces_old_chunks(indexer, rag_db_session):
    indexer.index(document_id=502, text=TEXT)
    indexer.index(document_id=502, text="Short new text.")

    assert chunk_count(rag_db_session, 502) == 1


def test_remove_deletes_chunks(indexer, rag_db_session):
    indexer.index(document_id=503, text=TEXT)

    removed = indexer.remove(503)

    assert removed > 0
    assert chunk_count(rag_db_session, 503) == 0


def test_empty_text_is_rejected(indexer):
    with pytest.raises(ValueError):
        indexer.index(document_id=504, text="")


def test_owner_can_index_document(client, indexer, rag_db_session):
    headers = login(client, "index_owner")

    document_id = client.post(
        "/documents/",
        json={"title": "guide", "category": "AI", "body": TEXT},
        headers=headers,
    ).json()["id"]

    response = client.post(
        f"/documents/{document_id}/index",
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["chunks"] == chunk_count(
        rag_db_session,
        document_id,
    )


def test_other_user_cannot_index_document(client, indexer):
    owner = login(client, "index_owner2")
    other = login(client, "index_other")

    document_id = client.post(
        "/documents/",
        json={"title": "guide", "category": "AI", "body": TEXT},
        headers=owner,
    ).json()["id"]

    response = client.post(
        f"/documents/{document_id}/index",
        headers=other,
    )

    assert response.status_code == 403


def test_document_without_body_cannot_be_indexed(client, indexer):
    headers = login(client, "index_owner3")

    document_id = client.post(
        "/documents/",
        json={"title": "empty", "category": "AI"},
        headers=headers,
    ).json()["id"]

    response = client.post(
        f"/documents/{document_id}/index",
        headers=headers,
    )

    assert response.status_code == 400


def test_index_requires_authentication(client, indexer):
    response = client.post("/documents/1/index")

    assert response.status_code == 401


def test_deleting_document_removes_its_chunks(
    client,
    indexer,
    rag_db_session,
):
    headers = login(client, "index_owner4")

    document_id = client.post(
        "/documents/",
        json={"title": "guide", "category": "AI", "body": TEXT},
        headers=headers,
    ).json()["id"]

    client.post(f"/documents/{document_id}/index", headers=headers)

    response = client.delete(
        f"/documents/{document_id}",
        headers=headers,
    )

    assert response.status_code == 200
    assert chunk_count(rag_db_session, document_id) == 0


def test_changing_body_removes_stale_chunks(
    client,
    indexer,
    rag_db_session,
):
    headers = login(client, "index_owner5")

    document_id = client.post(
        "/documents/",
        json={"title": "guide", "category": "AI", "body": TEXT},
        headers=headers,
    ).json()["id"]

    client.post(f"/documents/{document_id}/index", headers=headers)

    client.put(
        f"/documents/{document_id}",
        json={"body": "Completely new text."},
        headers=headers,
    )

    assert chunk_count(rag_db_session, document_id) == 0
