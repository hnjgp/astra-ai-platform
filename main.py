from fastapi import (
    FastAPI,
    Request,
    Depends
)

from fastapi.responses import JSONResponse

from database import engine

from models import Base

from exceptions import (
    DocumentNotFoundError,
    UserAlreadyExistsError,
    InvalidCredentialsError
)

from dependency import get_api_key

from router.document import router as document_router
from router.user import router as user_router


app = FastAPI(
    title="AI Backend API",
    version="1.0.0"
)


Base.metadata.create_all(
    bind=engine
)


app.include_router(
    document_router
)

app.include_router(
    user_router
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