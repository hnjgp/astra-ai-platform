import pytest
from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy import event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database import Base, get_db
from main import app


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
