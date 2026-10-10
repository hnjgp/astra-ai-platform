from llm.client import LLMClient
from services.ai_service import AIService


llm_client = LLMClient()
service = AIService(llm_client)


messages = [
    "Docker چیست؟",
    "من با Docker کار کردم ولی معماری Docker را عمیق توضیح بده",
    "برای مصاحبه AI Engineer چطور آماده شوم؟",
    "سلام، حالت چطوره؟",
]


for message in messages:

    print("=" * 60)
    print("MESSAGE:", message)

    answer = service.generate(message)

    print("\nANSWER:\n")
    print(answer)