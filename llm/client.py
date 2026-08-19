from urllib import response

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    OpenAI,
)

from config import OPENAI_API_KEY, OPENAI_MODEL
from exceptions import LLMError


class LLMClient:

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def generate(self, message: str) -> str:

        try:
            response = self.client.responses.create(
            model=OPENAI_MODEL,
            input=message,

            )

            print("MODEL:", response.model)
            print("USAGE:", response.usage)
            return response.output_text

        except AuthenticationError as exc:
            raise LLMError("LLM authentication failed") from exc

        except RateLimitError as exc:
            raise LLMError("LLM rate limit exceeded") from exc

        except BadRequestError as exc:
            raise LLMError("Invalid LLM request") from exc

        except APIConnectionError as exc:
            raise LLMError("Could not connect to LLM provider") from exc

        except APIStatusError as exc:
            raise LLMError("LLM provider returned an error") from exc