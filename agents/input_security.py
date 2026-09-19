import re


class InputSecurityChecker:
    INJECTION_PATTERNS = (
        r"\bignore\s+(all\s+)?previous\s+instructions\b",
        r"\bforget\s+(all\s+)?previous\s+instructions\b",
        r"\breveal\s+(your\s+)?system\s+prompt\b",
        r"\breveal\s+(your\s+)?instructions\b",
    )

    def check(
        self,
        message: str,
    ) -> None:
        if not message.strip():
            raise ValueError(
                "message must not be empty"
            )

        for pattern in self.INJECTION_PATTERNS:
            if re.search(
                pattern,
                message,
                flags=re.IGNORECASE,
            ):
                raise ValueError(
                    "potential prompt injection detected"
                )