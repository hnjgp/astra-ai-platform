"""
Metrics, request ids and structured (JSON) logging.

Metrics live in process memory and are exposed in Prometheus text
format at GET /metrics. With several worker processes each worker has
its own numbers; a real deployment would use prometheus_client or
OpenTelemetry with a shared backend.
"""

import json
import logging
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

from redaction import configured_secrets, redact_secrets


request_id_var: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)

Labels = tuple[tuple[str, str], ...]


def _labels(labels: dict[str, object]) -> Labels:
    return tuple(
        sorted(
            (key, str(value))
            for key, value in labels.items()
        )
    )


class Metrics:
    """
    Thread-safe counters and timers.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, dict[Labels, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._timer_sums: dict[str, dict[Labels, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._timer_counts: dict[str, dict[Labels, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def increment(
        self,
        name: str,
        value: float = 1.0,
        **labels,
    ) -> None:
        with self._lock:
            self._counters[name][_labels(labels)] += value

    def observe(
        self,
        name: str,
        seconds: float,
        **labels,
    ) -> None:
        key = _labels(labels)

        with self._lock:
            self._timer_sums[name][key] += seconds
            self._timer_counts[name][key] += 1

    def counter(self, name: str, **labels) -> float:
        with self._lock:
            return self._counters[name].get(_labels(labels), 0.0)

    def timer_count(self, name: str, **labels) -> int:
        with self._lock:
            return self._timer_counts[name].get(_labels(labels), 0)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timer_sums.clear()
            self._timer_counts.clear()

    def render_prometheus(self) -> str:
        lines: list[str] = []

        def label_text(labels: Labels) -> str:
            if not labels:
                return ""

            inner = ",".join(
                f'{key}="{value}"'
                for key, value in labels
            )

            return "{" + inner + "}"

        with self._lock:
            for name in sorted(self._counters):
                lines.append(f"# TYPE astra_{name}_total counter")

                for labels, value in sorted(
                    self._counters[name].items()
                ):
                    lines.append(
                        f"astra_{name}_total{label_text(labels)} {value:g}"
                    )

            for name in sorted(self._timer_sums):
                lines.append(f"# TYPE astra_{name}_seconds summary")

                for labels, total in sorted(
                    self._timer_sums[name].items()
                ):
                    count = self._timer_counts[name][labels]

                    lines.append(
                        f"astra_{name}_seconds_sum"
                        f"{label_text(labels)} {total:.6f}"
                    )
                    lines.append(
                        f"astra_{name}_seconds_count"
                        f"{label_text(labels)} {count}"
                    )

        return "\n".join(lines) + "\n"


metrics = Metrics()


@contextmanager
def timed(
    name: str,
    **labels,
) -> Iterator[None]:
    """
    Record duration and outcome (ok/error) of a block of work.
    """

    start = time.perf_counter()
    status = "ok"

    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        metrics.observe(
            name,
            time.perf_counter() - start,
            **labels,
        )
        metrics.increment(
            f"{name}s",
            status=status,
            **labels,
        )


# ------------------------------------------------------------
# Token usage and cost
# ------------------------------------------------------------

def record_llm_usage(
    operation: str,
    usage,
) -> None:
    """
    Count tokens of a Responses API call and add the estimated cost.
    """

    if usage is None:
        return

    from config import settings

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    metrics.increment(
        "llm_tokens",
        input_tokens,
        type="input",
        operation=operation,
    )
    metrics.increment(
        "llm_tokens",
        output_tokens,
        type="output",
        operation=operation,
    )

    cost = (
        input_tokens * settings.LLM_INPUT_COST_PER_1M_TOKENS
        + output_tokens * settings.LLM_OUTPUT_COST_PER_1M_TOKENS
    ) / 1_000_000

    metrics.increment(
        "llm_cost_usd",
        cost,
        operation=operation,
    )


def record_embedding_usage(usage) -> None:
    if usage is None:
        return

    from config import settings

    tokens = getattr(usage, "prompt_tokens", 0) or 0

    metrics.increment("embedding_tokens", tokens)
    metrics.increment(
        "embedding_cost_usd",
        tokens * settings.EMBEDDING_COST_PER_1M_TOKENS / 1_000_000,
    )


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """
    One JSON object per line, with secrets redacted.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = request_id_var.get()

        if request_id is not None:
            entry["request_id"] = request_id

        for key, value in getattr(record, "fields", {}).items():
            entry[key] = value

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        text = json.dumps(
            entry,
            ensure_ascii=False,
            default=str,
        )

        return redact_secrets(
            text,
            known_secrets=configured_secrets(),
        )


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger("astra")
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False


def log_event(
    logger: logging.Logger,
    message: str,
    level: int = logging.INFO,
    **fields,
) -> None:
    logger.log(
        level,
        message,
        extra={"fields": fields},
    )


# ------------------------------------------------------------
# HTTP request middleware
# ------------------------------------------------------------

_REQUEST_ID_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789-_"
)

http_logger = logging.getLogger("astra.http")


def _clean_request_id(value: str | None) -> str | None:
    # Client-provided ids end up in logs, so accept only short
    # alphanumeric values.
    if (
        value
        and len(value) <= 64
        and set(value) <= _REQUEST_ID_ALLOWED
    ):
        return value

    return None


class RequestContextMiddleware:
    """
    Request id, latency, status and route metrics for every request.

    Pure ASGI middleware, so streaming responses are not buffered.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from uuid import uuid4

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }

        request_id = (
            _clean_request_id(headers.get("x-request-id"))
            or uuid4().hex
        )

        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message):
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [
                    *message["headers"],
                    (b"x-request-id", request_id.encode()),
                ]

            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration = time.perf_counter() - start

            # The route template (/documents/{document_id}) keeps the
            # number of metric series small.
            route = scope.get("route")
            route_path = getattr(route, "path", "unmatched")

            metrics.observe(
                "http_request",
                duration,
                method=scope["method"],
                route=route_path,
            )
            metrics.increment(
                "http_requests",
                method=scope["method"],
                route=route_path,
                status=status_code,
            )

            log_event(
                http_logger,
                "request completed",
                method=scope["method"],
                route=route_path,
                status=status_code,
                duration_ms=round(duration * 1000, 2),
            )

            request_id_var.reset(token)
