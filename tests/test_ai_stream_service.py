from schemas import AIMessage
from services.ai_service import AIService


class FakeLLMClient:

    def chat_stream(self, messages):
        yield "Docker"
        yield " یک"
        yield " فناوری"
        yield " است."


def test_chat_stream():

    fake_llm = FakeLLMClient()

    service = AIService(
        llm_client=fake_llm
    )

    messages = [
        AIMessage(
            role="user",
            content="Docker چیست؟",
        )
    ]

    chunks = list(
        service.chat_stream(messages)
    )

    assert chunks == [
        "Docker",
        " یک",
        " فناوری",
        " است.",
    ]

    print("TEST: AIService Streaming PASS")


if __name__ == "__main__":
    test_chat_stream()