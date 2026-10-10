from llm.client import LLMClient
from prompts.router import route_message


llm_client = LLMClient()


messages = [
    "Docker چیست؟",
    "Docker رو از پایه برام توضیح بده",
    "من با Docker و Kubernetes کار کردم ولی معماری container orchestration رو عمیق توضیح بده",
    "برای مصاحبه AI Engineer چطور آماده شوم؟",
    "سلام، حالت چطوره؟",
]


for message in messages:

    print("=" * 60)
    print("MESSAGE:", message)

    route = route_message(
        message,
        llm_client,
    )

    print("\nROUTE:")
    print("intent:", route.intent)
    print("topic:", route.topic)
    print("level:", route.level)
    print("language:", route.language)