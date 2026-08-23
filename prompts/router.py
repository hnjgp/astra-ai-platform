from llm.client import LLMClient
from schemas import AIRoute


ROUTING_PROMPT = """
You are Astra's message router.

Analyze the user's message and classify it.

Return exactly four lines:

Category: <category>
Topic: <topic>
Level: <level>
Language: <language>

Category must be exactly one of:

- technical
- general
- career

Rules:

1. If Category is technical:
   - Extract the main technical concept as Topic.
   - Detect the user's apparent technical level.

2. If the user asks for a basic explanation
   or asks "what is X", use:
   Level: beginner

3. If the user mentions previous experience
   or says they already worked with the technology,
   use:
   Level: intermediate

4. If the user explicitly asks for deep architecture,
   internals, optimization, distributed systems,
   or advanced implementation details,
   use:
   Level: advanced

5. If Category is general:
   - Topic must be None.
   - Level must be None.

6. If Category is career:
   - Topic must be None.
   - Level must be None.

7. Detect the language of the user's message.
   Use the language name in English.

Examples:

User: "Docker چیست؟"
Category: technical
Topic: Docker
Level: beginner
Language: Persian

User: "Docker رو از پایه برام توضیح بده"
Category: technical
Topic: Docker
Level: beginner
Language: Persian

User: "من با Docker کار کردم ولی معماری Docker را عمیق توضیح بده"
Category: technical
Topic: Docker architecture
Level: advanced
Language: Persian

User: "من با Docker و Kubernetes کار کردم ولی معماری container orchestration رو عمیق توضیح بده"
Category: technical
Topic: container orchestration
Level: advanced
Language: Persian

User: "برای مصاحبه AI Engineer چطور آماده شوم؟"
Category: career
Topic: None
Level: None
Language: Persian

User: "سلام، حالت چطوره؟"
Category: general
Topic: None
Level: None
Language: Persian

Important:

- Return exactly four lines.
- Do not add explanations.
- Do not use Markdown.

Now analyze this message:

User: {message}

Category: <category>
Topic: <topic>
Level: <level>
Language: <language>
""".strip()


def route_message(message: str, llm_client: LLMClient) -> AIRoute:

    prompt = ROUTING_PROMPT.format(message=message)

    result = llm_client.generate(
        message=prompt,
        instructions=(
            "Classify the message and extract "
            "topic, level, and language. "
            "Return exactly four lines "
            "in the requested format."
        ),
    )

    lines = [
        line.strip()
        for line in result.strip().splitlines()
        if line.strip()
    ]

    category = "general"
    topic = None
    level = None
    language = None

    for line in lines:

        if line.lower().startswith("category:"):
            category = (
                line.split(":", 1)[1]
                .strip()
                .lower()
            )

        elif line.lower().startswith("topic:"):
            extracted_topic = (
                line.split(":", 1)[1]
                .strip()
            )

            if extracted_topic.lower() != "none":
                topic = extracted_topic

        elif line.lower().startswith("level:"):
            extracted_level = (
                line.split(":", 1)[1]
                .strip()
                .lower()
            )

            if extracted_level != "none":
                level = extracted_level

        elif line.lower().startswith("language:"):
            extracted_language = (
                line.split(":", 1)[1]
                .strip()
            )

            if extracted_language.lower() != "none":
                language = extracted_language

    if category not in {
        "technical",
        "general",
        "career",
    }:
        category = "general"

    if category != "technical":
        topic = None
        level = None

    if level not in {
        None,
        "beginner",
        "intermediate",
        "advanced",
    }:
        level = None

    return AIRoute(
        intent=category,
        topic=topic,
        level=level,
        language=language,
    )