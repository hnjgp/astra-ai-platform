class FakeLLMClient:

    def __init__(self):
        self.calls = []

    def generate(self, message: str, instructions: str) -> str:

        self.calls.append(
            {
                "message": message,
                "instructions": instructions,
            }
        )

        return "FAKE LLM RESPONSE"

    def chat(self, messages) -> str:
        self.calls.append(
            {
                "messages": messages,
            }
        )

        return "FAKE CHAT RESPONSE"