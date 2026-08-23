from services.ai_service import AIService


class FakeLLMClient:

    def generate(self, message: str, instructions: str) -> str:
        raise RuntimeError("LLM service failed")


service = AIService(FakeLLMClient())


try:
    result = service.generate("Docker چیست؟")

    print("RESULT:")
    print(result)

except Exception as e:
    print("ERROR:")
    print(type(e).__name__)
    print(str(e))