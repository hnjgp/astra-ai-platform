from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from models import Base


from config import DATABASE_URL


engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
)


@event.listens_for(engine, "connect")
def enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
    """Make SQLite enforce the same relationship rules as production DBs."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
