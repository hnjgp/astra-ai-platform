# llm/client.py

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

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
from observability import log_event, record_llm_usage, timed
from prompts.ai import ASTRA_SYSTEM_PROMPT
from schemas import AIMessage


logger = logging.getLogger("astra.llm")


@contextmanager
def provider_errors() -> Iterator[None]:
    """
    Translate provider exceptions into LLMError so the rest of the
    application does not depend on the provider SDK.
    """

    try:
        yield

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


class LLMClient:

    def __init__(
        self,
        client: OpenAI | None = None,
        model: str = OPENAI_MODEL,
    ):
        self.client = client or OpenAI(
            api_key=OPENAI_API_KEY
        )
        self.model = model

    def _create(
        self,
        operation: str,
        **kwargs: Any,
    ):
        with timed(
            "llm_call",
            operation=operation,
        ):
            with provider_errors():
                response = self.client.responses.create(
                    model=self.model,
                    **kwargs,
                )

        record_llm_usage(
            operation,
            getattr(response, "usage", None),
        )

        log_event(
            logger,
            "llm call completed",
            operation=operation,
            model=getattr(response, "model", self.model),
        )

        return response

    def generate(
        self,
        message: str,
        instructions: str,
    ) -> str:

        return self._create(
            "generate",
            instructions=instructions,
            input=message,
        ).output_text

    def chat(
        self,
        messages: list[AIMessage],
    ) -> str:

        return self._create(
            "chat",
            instructions=ASTRA_SYSTEM_PROMPT,
            input=[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
        ).output_text

    def chat_stream(
        self,
        messages: list[AIMessage],
    ) -> Iterator[str]:

        with timed(
            "llm_call",
            operation="chat_stream",
        ):
            with provider_errors():
                stream = self.client.responses.create(
                    model=self.model,
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

                    elif event.type == "response.completed":
                        record_llm_usage(
                            "chat_stream",
                            getattr(event.response, "usage", None),
                        )

    def generate_with_tools(
        self,
        message: str | list[dict[str, Any]],
        tools: list[dict],
        instructions: str | None = None,
        previous_response_id: str | None = None,
    ):

        return self._create(
            "generate_with_tools",
            instructions=(
                instructions
                or ASTRA_SYSTEM_PROMPT
            ),
            input=message,
            tools=tools,
            previous_response_id=previous_response_id,
        )
