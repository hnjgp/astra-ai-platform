from typing import Callable, Iterator

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from config import OPENAI_API_KEY, OPENAI_MODEL
from exceptions import LLMError
from prompts.ai import ASTRA_SYSTEM_PROMPT
from schemas import AIMessage


class LLMClient:

    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    # ============================================================
    # Basic Generate
    # ============================================================

    def generate(
        self,
        message: str,
        instructions: str,
    ) -> str:

        try:

            response = self.client.responses.create(
                model=OPENAI_MODEL,
                instructions=instructions,
                input=message,
            )

            print("MODEL:", response.model)
            print("USAGE:", response.usage)

            return response.output_text

        except AuthenticationError as exc:

            raise LLMError(
                "LLM authentication failed"
            ) from exc

        except RateLimitError as exc:

            raise LLMError(
                "LLM rate limit exceeded"
            ) from exc

        except BadRequestError as exc:

            raise LLMError(
                "Invalid LLM request"
            ) from exc

        except APIConnectionError as exc:

            raise LLMError(
                "Could not connect to LLM provider"
            ) from exc

        except APIStatusError as exc:

            raise LLMError(
                "LLM provider returned an error"
            ) from exc

    # ============================================================
    # Chat
    # ============================================================

    def chat(
        self,
        messages: list[AIMessage],
    ) -> str:

        try:

            response = self.client.responses.create(
                model=OPENAI_MODEL,
                instructions=ASTRA_SYSTEM_PROMPT,
                input=[
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in messages
                ],
            )

            print("MODEL:", response.model)
            print("USAGE:", response.usage)

            return response.output_text

        except AuthenticationError as exc:

            raise LLMError(
                "LLM authentication failed"
            ) from exc

        except RateLimitError as exc:

            raise LLMError(
                "LLM rate limit exceeded"
            ) from exc

        except BadRequestError as exc:

            raise LLMError(
                "Invalid LLM request"
            ) from exc

        except APIConnectionError as exc:

            raise LLMError(
                "Could not connect to LLM provider"
            ) from exc

        except APIStatusError as exc:

            raise LLMError(
                "LLM provider returned an error"
            ) from exc

    # ============================================================
    # Streaming Chat
    # ============================================================

    def chat_stream(
        self,
        messages: list[AIMessage],
    ) -> Iterator[str]:

        try:

            stream = self.client.responses.create(
                model=OPENAI_MODEL,
                instructions=ASTRA_SYSTEM_PROMPT,
                input=[
                    {
                        "role": message.role,
                        "content": message.content,
                    }
                    for message in messages
                ],
                stream=True,
            )

            for event in stream:

                if event.type == "response.output_text.delta":

                    yield event.delta

        except AuthenticationError as exc:

            raise LLMError(
                "LLM authentication failed"
            ) from exc

        except RateLimitError as exc:

            raise LLMError(
                "LLM rate limit exceeded"
            ) from exc

        except BadRequestError as exc:

            raise LLMError(
                "Invalid LLM request"
            ) from exc

        except APIConnectionError as exc:

            raise LLMError(
                "Could not connect to LLM provider"
            ) from exc

        except APIStatusError as exc:

            raise LLMError(
                "LLM provider returned an error"
            ) from exc

    # ============================================================
    # Generate With Tools
    # ============================================================

    def generate_with_tools(
        self,
        message: str,
        tools: list[dict],
        instructions: str | None = None,
    ):

        try:

            response = self.client.responses.create(
                model=OPENAI_MODEL,
                instructions=(
                    instructions
                    or ASTRA_SYSTEM_PROMPT
                ),
                input=message,
                tools=tools,
            )

            return response

        except AuthenticationError as exc:

            raise LLMError(
                "LLM authentication failed"
            ) from exc

        except RateLimitError as exc:

            raise LLMError(
                "LLM rate limit exceeded"
            ) from exc

        except BadRequestError as exc:

            raise LLMError(
                "Invalid LLM request"
            ) from exc

        except APIConnectionError as exc:

            raise LLMError(
                "Could not connect to LLM provider"
            ) from exc

        except APIStatusError as exc:

            raise LLMError(
                "LLM provider returned an error"
            ) from exc

    # ============================================================
    # Generate With Tool Execution
    # ============================================================

    def generate_with_tool_execution(
        self,
        message: str,
        tools: list[dict],
        tool_executor: Callable,
        max_tool_rounds: int = 5,
        instructions: str | None = None,
    ) -> str:

        try:

            response = self.client.responses.create(
                model=OPENAI_MODEL,
                instructions=(
                    instructions
                    or ASTRA_SYSTEM_PROMPT
                ),
                input=message,
                tools=tools,
            )

            for round_number in range(
                1,
                max_tool_rounds + 1,
            ):

                tool_calls = [
                    item
                    for item in response.output
                    if item.type == "function_call"
                ]

                # ------------------------------------------------
                # Model finished without requesting another tool
                # ------------------------------------------------

                if not tool_calls:

                    return response.output_text

                print()
                print(
                    f"TOOL ROUND: {round_number}"
                )

                tool_outputs = []

                # ------------------------------------------------
                # Execute requested tools
                # ------------------------------------------------

                for tool_call in tool_calls:

                    print(
                        "TOOL CALL:",
                        tool_call.name,
                    )

                    print(
                        "ARGUMENTS:",
                        tool_call.arguments,
                    )

                    result = tool_executor(
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                    )

                    print(
                        "TOOL RESULT:",
                        result,
                    )

                    # ------------------------------------------------
                    # ToolResult -> JSON
                    #
                    # The LLM receives a stable JSON contract instead
                    # of Python object representation.
                    # ------------------------------------------------

                    tool_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": (
                                result.model_dump_json()
                            ),
                        }
                    )

                # ------------------------------------------------
                # Send tool results back to the model
                #
                # Keep tools available so the model can request
                # another tool in the next round.
                # ------------------------------------------------

                response = self.client.responses.create(
                    model=OPENAI_MODEL,
                    instructions=(
                        instructions
                        or ASTRA_SYSTEM_PROMPT
                    ),
                    input=tool_outputs,
                    previous_response_id=response.id,
                    tools=tools,
                )

            # ----------------------------------------------------
            # Safety limit
            # ----------------------------------------------------

            raise LLMError(
                "Maximum tool execution rounds exceeded"
            )

        except AuthenticationError as exc:

            raise LLMError(
                "LLM authentication failed"
            ) from exc

        except RateLimitError as exc:

            raise LLMError(
                "LLM rate limit exceeded"
            ) from exc

        except BadRequestError as exc:

            raise LLMError(
                "Invalid LLM request"
            ) from exc

        except APIConnectionError as exc:

            raise LLMError(
                "Could not connect to LLM provider"
            ) from exc

        except APIStatusError as exc:

            raise LLMError(
                "LLM provider returned an error"
            ) from exc