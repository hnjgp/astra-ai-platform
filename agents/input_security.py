import re
import unicodedata


class InputSecurityChecker:
    # A first layer for obvious attacks only; it is not a complete
    # defence. Tool authorization, approval and output redaction are
    # the layers that limit damage when an injection gets through.
    INJECTION_PATTERNS = (
        r"\bignore\s+(all\s+)?previous\s+instructions\b",
        r"\bforget\s+(all\s+)?previous\s+instructions\b",
        r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        r"\breveal\s+(your\s+)?system\s+prompt\b",
        r"\breveal\s+(your\s+)?instructions\b",
        r"\byou\s+are\s+now\s+in\s+developer\s+mode\b",
        r"دستور(ات|العمل(‌|\s)?ها(ی)?)\s+قبلی\s+را\s+نادیده\s+بگیر",
        r"پرامپت\s+سیستم(ی)?\s+(خود|ات|ت)?\s*را\s+(نشان|افشا)",
    )

    def check(
        self,
        message: str,
    ) -> None:
        if not message.strip():
            raise ValueError(
                "message must not be empty"
            )

        if self._has_hidden_characters(message):
            raise ValueError(
                "message contains control or invisible characters"
            )

        # Normalise so look-alike Unicode forms cannot bypass patterns.
        normalized = unicodedata.normalize(
            "NFKC",
            message,
        )

        for pattern in self.INJECTION_PATTERNS:
            if re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                raise ValueError(
                    "potential prompt injection detected"
                )

    @staticmethod
    def _has_hidden_characters(message: str) -> bool:
        for character in message:
            if character in "\n\r\t":
                continue

            category = unicodedata.category(character)

            # Cc = control characters. Cf = format characters such as
            # zero-width spaces, except the zero-width non-joiner that
            # Persian text uses between word parts.
            if category == "Cc":
                return True

            if category == "Cf" and character != "‌":
                return True

        return False
