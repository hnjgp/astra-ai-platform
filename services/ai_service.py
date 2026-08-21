from llm.client import LLMClient
from prompts.ai import ASTRA_SYSTEM_PROMPT, build_teacher_prompt
from prompts.router import route_message
from schemas import AIMessage


class AIService:

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    def generate(self, message: str) -> str:
        route = route_message(message, self.llm_client)

        if route.intent == "technical":
            instructions = build_teacher_prompt(message)
        else:
            instructions = ASTRA_SYSTEM_PROMPT

        return self.llm_client.generate(
            message=message,
            instructions=instructions,
        )

    def chat(self, messages: list[AIMessage]) -> str:
        return self.llm_client.chat(messages)