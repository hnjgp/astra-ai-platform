from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from models import Base


from config import DATABASE_URL


IS_SQLITE = DATABASE_URL.startswith("sqlite")


engine = create_engine(
    DATABASE_URL,
    # SQLite connections are used from FastAPI's thread pool.
    connect_args=(
        {"check_same_thread": False}
        if IS_SQLITE
        else {}
    ),
    # Detect connections dropped by the server before using them.
    pool_pre_ping=not IS_SQLITE,
)


if IS_SQLITE:

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


def init_app_database() -> None:
    """Create tables and add columns/indexes introduced later.

    Production deployments should use versioned migrations (Alembic);
    this keeps an existing developer database usable.
    """
    Base.metadata.create_all(bind=engine)

    if not IS_SQLITE:
        return

    with engine.begin() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(documents)")}
        additions = {
            "body": "VARCHAR",
            "is_private": "BOOLEAN NOT NULL DEFAULT 0",
            "owner_id": "INTEGER",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.exec_driver_sql(f"ALTER TABLE documents ADD COLUMN {name} {definition}")

        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_documents_owner_id ON documents (owner_id)"
        )
