from schemas import AIMessage
from prompts.ai import (
    ASTRA_SYSTEM_PROMPT,
    build_teacher_prompt,
)
from prompts.career import build_career_prompt
from prompts.router import route_message
from exceptions import LLMError


class AIService:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(self, message: str) -> str:
        try:
            route = route_message(
                message,
                self.llm_client,
            )

            if route.intent == "technical":
                instructions = build_teacher_prompt(
                    topic=route.topic or "Unknown",
                    level=route.level or "beginner",
                    language=route.language or "Persian",
                )

            elif route.intent == "career":
                instructions = build_career_prompt()

            else:
                instructions = ASTRA_SYSTEM_PROMPT

            return self.llm_client.generate(
                message=message,
                instructions=instructions,
            )

        except LLMError:
            print(
                "AIService error: LLMError"
            )
            raise

        except Exception as exc:
            print(
                f"AIService error: "
                f"{type(exc).__name__}: {exc}"
            )

            return "متأسفانه در پردازش درخواست شما مشکلی پیش آمد."

    def chat(self, messages: list[AIMessage]) -> str:
        try:
            return self.llm_client.chat(
                messages
            )

        except Exception as exc:
            print(
                f"AIService error: "
                f"{type(exc).__name__}: {exc}"
            )

            raise