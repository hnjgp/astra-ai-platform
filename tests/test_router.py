from llm.client import LLMClient
from prompts.router import route_message


llm_client = LLMClient()


messages = [
    "Docker چیست؟",
    "Docker رو از پایه برام توضیح بده",
    "یه توضیح ساده درباره Redis می‌خوام",
    "FastAPI چطور کار می‌کند؟",
    "سلام، حالت چطوره؟",
    "برای مصاحبه AI Engineer چطور آماده شوم؟",
    "برای رزومه AI Engineer چه چیزهایی بنویسم؟",
]


for message in messages:
    result = route_message(message, llm_client)

    print("=" * 50)
    print("MESSAGE:", message)
    print("INTENT:", result.intent)
    print("TOPIC:", result.topic)