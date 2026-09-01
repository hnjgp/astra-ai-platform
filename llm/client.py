from typing import Any, Iterator

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

            print(
                "MODEL:",
                response.model,
            )

            print(
                "USAGE:",
                response.usage,
            )

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

            print(
                "MODEL:",
                response.model,
            )

            print(
                "USAGE:",
                response.usage,
            )

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

                if event.type == (
                    "response.output_text.delta"
                ):
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

    def generate_with_tools(
        self,
        message: str | list[dict[str, Any]],
        tools: list[dict],
        instructions: str | None = None,
        previous_response_id: str | None = None,
    ):

        try:

            kwargs = {
                "model": OPENAI_MODEL,
                "instructions": (
                    instructions
                    or ASTRA_SYSTEM_PROMPT
                ),
                "input": message,
                "tools": tools,
            }

            if previous_response_id is not None:
                kwargs["previous_response_id"] = (
                    previous_response_id
                )

            response = self.client.responses.create(
                **kwargs
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