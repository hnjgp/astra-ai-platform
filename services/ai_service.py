from llm.client import LLMClient


class AIService:

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate(self, message: str) -> str:
        return self.llm_client.generate(message)