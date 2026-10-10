import re
from collections.abc import Iterable


REDACTED = "[REDACTED]"

SECRET_PATTERNS = (
    # OpenAI-style API keys
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    # JSON Web Tokens
    re.compile(r"\beyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}"),
    # AWS access key ids
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # PEM private keys
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
        r".*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    # Credentials inside connection strings: scheme://user:password@
    re.compile(r"(?<=://)[^\s:/@]+:[^\s@/]+(?=@)"),
)


def redact_secrets(
    text: str,
    known_secrets: Iterable[str | None] = (),
) -> str:
    """
    Replace secrets in text with [REDACTED].

    known_secrets are exact values (for example from configuration)
    that must never leave the system.
    """

    for secret in known_secrets:
        # Very short values would redact ordinary words.
        if secret and len(secret) >= 8:
            text = text.replace(secret, REDACTED)

    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)

    return text


def configured_secrets() -> list[str | None]:
    from config import settings

    return [
        settings.SECRET_KEY,
        settings.OPENAI_API_KEY,
        settings.INTERNAL_API_KEY,
    ]
