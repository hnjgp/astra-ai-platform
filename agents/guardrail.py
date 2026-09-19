class AgentGuardrail:
    def __init__(
        self,
        max_input_length: int = 10_000,
        max_output_length: int = 20_000,
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