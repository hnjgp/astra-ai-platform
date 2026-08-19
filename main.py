from fastapi import (
    FastAPI,
    Request,
    Depends
)


from exceptions import LLMError

from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from time import monotonic

from database import DATABASE_URL, engine

from models import Base

from exceptions import (
    DocumentNotFoundError,
    UserAlreadyExistsError,
    InvalidCredentialsError
)

from dependency import get_api_key

from router.document import router as document_router
from router.user import router as user_router
from router.ai import router as ai_router





app = FastAPI(
    title="AI Backend API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """A small in-process guard for login and write endpoints.

    A shared store such as Redis should replace this in a multi-process
    deployment; keeping it here makes the local API safe by default.
    """
    window_seconds = 60
    max_requests = 120
    requests: dict[str, list[float]] = {}

    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        now = monotonic()
        client = request.client.host if request.client else "unknown"
        key = f"{client}:{request.url.path}"
        recent = [stamp for stamp in self.requests.get(key, []) if now - stamp < self.window_seconds]
        if len(recent) >= self.max_requests:
            return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
        recent.append(now)
        self.requests[key] = recent
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


Base.metadata.create_all(
    bind=engine
)


def migrate_sqlite_schema() -> None:
    """Add columns introduced after the first local SQLite release.

    Production deployments should use versioned migrations; this keeps an
    existing developer database usable while the project remains SQLite based.
    """
    if not DATABASE_URL.startswith("sqlite"):
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


migrate_sqlite_schema()


# Routers

app.include_router(
    user_router
)

app.include_router(
    document_router
)



# -------------------------
# Exception Handlers
# -------------------------


@app.exception_handler(
    DocumentNotFoundError
)
async def document_not_found_handler(
    request: Request,
    exc: DocumentNotFoundError
):

    return JSONResponse(
        status_code=404,
        content={
            "detail": "document not found"
        }
    )



@app.exception_handler(
    UserAlreadyExistsError
)
async def user_already_exists_handler(
    request: Request,
    exc: UserAlreadyExistsError
):

    return JSONResponse(
        status_code=400,
        content={
            "detail": "username already exists"
        }
    )



@app.exception_handler(
    InvalidCredentialsError
)
async def invalid_credentials_handler(
    request: Request,
    exc: InvalidCredentialsError
):

    return JSONResponse(
        status_code=401,
        content={
            "detail": "incorrect username or password"
        },
        headers={
            "WWW-Authenticate": "Bearer"
        }
    )



# -------------------------
# Basic endpoints
# -------------------------


@app.get("/health")
def health():

    return {
        "status": "ok"
    }



@app.get("/auth-test")
def auth_test(
    api_key=Depends(get_api_key)
):

    return {
        "api_key": api_key
    }


@app.exception_handler(LLMError)
async def llm_error_handler(
    request: Request,
    exc: LLMError,
):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "AI service is temporarily unavailable"
        },
    )