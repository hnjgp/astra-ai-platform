from llm.client import LLMClient
from prompts.ai import build_teacher_prompt


class TechnicalService:

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate(self, message: str, topic: str) -> str:
        instructions = build_teacher_prompt(topic)

        return self.llm_client.generate(
            message=message,
            instructions=instructions,
        )