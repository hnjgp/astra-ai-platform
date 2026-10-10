from collections.abc import Iterable

from agents.input_security import InputSecurityChecker
from agents.redaction import configured_secrets, redact_secrets


class AgentGuardrail:
    def __init__(
        self,
        max_input_length: int = 10_000,
        max_output_length: int = 20_000,
        input_security_checker: InputSecurityChecker | None = None,
        known_secrets: Iterable[str | None] | None = None,
    ):
        if max_input_length <= 0:
            raise ValueError(
                "max_input_length must be greater than zero"
            )

        if max_output_length <= 0:
            raise ValueError(
                "max_output_length must be greater than zero"
            )

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        # None = the secrets from the application configuration.
        self.known_secrets = (
            list(known_secrets)
            if known_secrets is not None
            else None
        )
        self.input_security_checker = (
            input_security_checker
            or InputSecurityChecker()
        )

    def validate_input(
        self,
        message: str,
    ) -> None:
        if not message.strip():
            raise ValueError(
                "message must not be empty"
            )

        if len(message) > self.max_input_length:
            raise ValueError(
                "message exceeds maximum input length"
            )

        self.input_security_checker.check(
            message
        )

    def validate_output(
        self,
        output: str,
    ) -> None:
        if not output.strip():
            raise ValueError(
                "output must not be empty"
            )

        if len(output) > self.max_output_length:
            raise ValueError(
                "output exceeds maximum output length"
            )
    def sanitize_output(
        self,
        output: str,
    ) -> str:
        """
        Validate the final answer and remove secrets from it.
        """

        self.validate_output(output)

        return redact_secrets(
            output,
            known_secrets=(
                self.known_secrets
                if self.known_secrets is not None
                else configured_secrets()
            ),
        )
