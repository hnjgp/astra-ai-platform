import math
from time import monotonic

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from security import InvalidTokenError, decode_access_token


AI_PATH_PREFIX = "/ai/"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limits kept in process memory.

    - General endpoints: per client IP and path.
    - AI endpoints (expensive model calls): one shared budget per user,
      or per IP when the request has no valid token.

    A shared store such as Redis should replace this in a multi-process
    deployment; keeping it here makes the local API safe by default.
    """

    window_seconds = 60
    max_requests = settings.RATE_LIMIT_PER_MINUTE
    max_ai_requests = settings.AI_RATE_LIMIT_PER_MINUTE
    cleanup_every = 1_000

    requests: dict[str, list[float]] = {}
    _seen = 0

    def _client_ip(self, request) -> str:
        return request.client.host if request.client else "unknown"

    def _ai_identity(self, request) -> str:
        authorization = request.headers.get("authorization", "")

        scheme, _, token = authorization.partition(" ")

        if scheme.lower() == "bearer" and token:
            try:
                payload = decode_access_token(token)
                return f"user:{payload['user_id']}"
            except InvalidTokenError:
                pass

        return f"ip:{self._client_ip(request)}"

    def _key_and_limit(self, request) -> tuple[str, int]:
        path = request.url.path

        if path.startswith(AI_PATH_PREFIX):
            return (
                f"ai:{self._ai_identity(request)}",
                self.max_ai_requests,
            )

        return (
            f"{self._client_ip(request)}:{path}",
            self.max_requests,
        )

    def _cleanup(self, now: float) -> None:
        cls = type(self)
        cls._seen += 1

        if cls._seen % self.cleanup_every:
            return

        for key in list(self.requests):
            if all(
                now - stamp >= self.window_seconds
                for stamp in self.requests[key]
            ):
                del self.requests[key]

    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        now = monotonic()
        self._cleanup(now)

        key, limit = self._key_and_limit(request)

        recent = [
            stamp
            for stamp in self.requests.get(key, [])
            if now - stamp < self.window_seconds
        ]

        if len(recent) >= limit:
            retry_after = math.ceil(
                self.window_seconds - (now - recent[0])
            )

            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded"},
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        recent.append(now)
        self.requests[key] = recent

        return await call_next(request)
