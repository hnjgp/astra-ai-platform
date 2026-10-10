import logging
from contextlib import ExitStack, asynccontextmanager

from fastapi import (
    FastAPI,
    Request,
    Depends
)


from exceptions import LLMError
from rag.exceptions import RAGError

from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from database import check_app_database, init_app_database
from rag.database import check_rag_database, init_rag_database


from exceptions import (
    DocumentNotFoundError,
    UserAlreadyExistsError,
    InvalidCredentialsError
)

from dependency import get_api_key
from rate_limit import RateLimitMiddleware
from config import settings
from observability import (
    RequestContextMiddleware,
    metrics,
    setup_logging,
)

from router.document import router as document_router
from router.user import router as user_router
from router.ai import router as ai_router
from router.ai import configure_agent_checkpointer
from agents.checkpointer import postgres_checkpointer





logger = logging.getLogger("astra.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_rag_database()
    except Exception:
        # The API can still serve non-RAG endpoints; /ready reports
        # the knowledge base as unavailable until the database is up.
        logger.exception("RAG database initialisation failed")

    with ExitStack() as stack:
        if settings.AGENT_CHECKPOINTER == "postgres":
            configure_agent_checkpointer(
                stack.enter_context(
                    postgres_checkpointer(
                        settings.RAG_DATABASE_URL
                    )
                )
            )

        yield

        configure_agent_checkpointer(None)


app = FastAPI(
    title="Astra AI Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router)


app.add_middleware(RateLimitMiddleware)

# Added last so it is the outermost middleware and also measures
# requests rejected by the rate limiter.
app.add_middleware(RequestContextMiddleware)

setup_logging(settings.LOG_LEVEL)


init_app_database()


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
    """
    Liveness: the process is up. Does not touch dependencies.
    """

    return {
        "status": "ok"
    }


@app.get("/ready")
def ready():
    """
    Readiness: the dependencies needed to serve traffic respond.
    """

    checks = {}

    for name, check in (
        ("database", check_app_database),
        ("rag_database", check_rag_database),
    ):
        try:
            check()
            checks[name] = "ok"
        except Exception:
            logger.exception("readiness check failed: %s", name)
            checks[name] = "unavailable"

    is_ready = all(
        status == "ok"
        for status in checks.values()
    )

    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "checks": checks,
        },
    )



@app.get(
    "/metrics",
    response_class=PlainTextResponse,
)
def read_metrics(
    api_key=Depends(get_api_key)
):
    """
    Prometheus metrics: requests, latency, model/tool/RAG calls,
    token usage and estimated cost. Protected by the internal API key.
    """

    return metrics.render_prometheus()


@app.get("/auth-test")
def auth_test(
    api_key=Depends(get_api_key)
):

    # Never echo the key back to the caller.
    return {
        "authenticated": True
    }


@app.exception_handler(RAGError)
async def rag_error_handler(
    request: Request,
    exc: RAGError,
):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "knowledge base is temporarily unavailable"
        },
    )


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