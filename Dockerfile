FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 astra

# Dependencies first, so code changes do not reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=astra:astra . .

# Uploaded documents are written here (mounted as a volume in compose).
RUN mkdir -p /app/documents && chown astra:astra /app/documents

USER astra

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
