import os
from pathlib import Path

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def use_test_rag_database() -> None:
    """
    Point the RAG database at "<name>_test" before the app is imported.

    Tests delete chunks, so they must never run against the development
    knowledge base. TEST_RAG_DATABASE_URL overrides the derived URL.
    """

    url = os.getenv("TEST_RAG_DATABASE_URL")

    if url is None:
        base_url = os.getenv("RAG_DATABASE_URL") or dotenv_values(
            Path(__file__).resolve().parents[1] / ".env"
        ).get("RAG_DATABASE_URL")

        if not base_url:
            return

        base = make_url(base_url)

        if base.database.endswith("_test"):
            url = base_url
        else:
            test_url = base.set(database=f"{base.database}_test")
            url = test_url.render_as_string(hide_password=False)

            try:
                # The maintenance database always exists on a server.
                admin_engine = create_engine(
                    base.set(database="postgres"),
                    isolation_level="AUTOCOMMIT",
                )

                with admin_engine.connect() as connection:
                    exists = connection.execute(
                        text("SELECT 1 FROM pg_database WHERE datname = :name"),
                        {"name": test_url.database},
                    ).scalar()

                    if not exists:
                        connection.execute(
                            text(f'CREATE DATABASE "{test_url.database}"')
                        )

                admin_engine.dispose()

            except Exception:
                # Database not reachable: integration tests will fail
                # with a clear connection error; unit tests still run.
                pass

    os.environ["RAG_DATABASE_URL"] = url


use_test_rag_database()


from database import Base, get_db  # noqa: E402
from main import app  # noqa: E402

from rag.database import RAGSessionLocal, init_rag_database  # noqa: E402
from rag.models import DocumentChunk  # noqa: E402


SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
        "check_same_thread": False
    },
    poolclass=StaticPool,
)


TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


@event.listens_for(engine, "connect")
def enable_foreign_keys(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# The administrative test helper imports SessionLocal at runtime.  Point it at
# the same isolated database used by the API dependency override.
import database

database.SessionLocal = TestingSessionLocal


INTEGRATION_MODULES = {
    "test_rag_database.py",
    "test_database_architecture.py",
    "test_readiness.py",
    "test_document_indexing.py",
    "test_e2e.py",
}


def pytest_collection_modifyitems(config, items):
    # Integration tests need the pgvector database:
    # run only unit tests with  -m "not integration".
    for item in items:
        if (
            "rag_db_session" in item.fixturenames
            or item.path.name in INTEGRATION_MODULES
        ):
            item.add_marker(pytest.mark.integration)

    # Live tests call the real LLM provider; they are opt-in so the
    # default suite stays deterministic and runs without network access.
    if os.getenv("ASTRA_RUN_LIVE_TESTS") == "1":
        return

    skip_live = pytest.mark.skip(
        reason="live test: set ASTRA_RUN_LIVE_TESTS=1 to run"
    )

    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def isolated_upload_directory(tmp_path, monkeypatch):
    import services.document_upload_service as upload_service

    monkeypatch.setattr(
        upload_service,
        "UPLOAD_DIRECTORY",
        tmp_path / "uploads",
    )


@pytest.fixture(autouse=True)
def setup_database():

    from main import RateLimitMiddleware
    from router.user import _failed_logins

    RateLimitMiddleware.requests.clear()
    _failed_logins.clear()

    Base.metadata.create_all(
        bind=engine
    )

    yield

    Base.metadata.drop_all(
        bind=engine
    )


@pytest.fixture(scope="session")
def rag_database():
    init_rag_database()


@pytest.fixture
def rag_db_session(rag_database):
    db = RAGSessionLocal()

    try:
        db.query(DocumentChunk).delete()
        db.commit()

        yield db

    finally:
        db.rollback()

        db.query(DocumentChunk).delete()
        db.commit()

        db.close()


def override_get_db():

    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client