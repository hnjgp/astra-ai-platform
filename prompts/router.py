from llm.client import LLMClient
from schemas import AIRoute


ROUTING_PROMPT = """
You are Astra's message router.

Classify the user's message into exactly one category:

- technical
- general
- career

Also extract the main topic when the message is technical.

Examples:

User: "Docker چیست؟"
Category: technical
Topic: Docker

User: "Docker رو از پایه برام توضیح بده"
Category: technical
Topic: Docker

User: "یه توضیح ساده درباره Redis می‌خوام"
Category: technical
Topic: Redis

User: "FastAPI چطور کار می‌کند؟"
Category: technical
Topic: FastAPI

User: "سلام، حالت چطوره؟"
Category: general
Topic: None

User: "امروز چه خبر؟"
Category: general
Topic: None

User: "برای مصاحبه AI Engineer چطور آماده شوم؟"
Category: career
Topic: None

User: "برای رزومه AI Engineer چه چیزهایی بنویسم؟"
Category: career
Topic: None

Rules:

- Category must be exactly one of: technical, general, career.
- If Category is technical, extract the main technical concept.
- If Category is general or career, Topic must be None.
- Return exactly two lines.
- Do not add explanations.
- Do not use Markdown.

Now analyze this message:

User: {message}

Category: <category>
Topic: <topic>
""".strip()


def route_message(message: str, llm_client: LLMClient) -> AIRoute:
    prompt = ROUTING_PROMPT.format(message=message)

    result = llm_client.generate(
        message=prompt,
        instructions=(
            "Classify the message and extract the topic. "
            "Return exactly two lines in the requested format."
        ),
    )

    lines = [
        line.strip()
        for line in result.strip().splitlines()
        if line.strip()
    ]

    category = "general"
    topic = None

    for line in lines:
        if line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip().lower()

        elif line.lower().startswith("topic:"):
            extracted_topic = line.split(":", 1)[1].strip()

            if extracted_topic.lower() != "none":
                topic = extracted_topic

    if category not in {"technical", "general", "career"}:
        category = "general"

    if category != "technical":
        topic = None

    return AIRoute(
        intent=category,
        topic=topic,
    )