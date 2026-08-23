from prompts.ai import build_teacher_prompt


prompt = build_teacher_prompt(
    topic="PostgreSQL indexing",
    level="advanced",
    language="English",
)

print(prompt)