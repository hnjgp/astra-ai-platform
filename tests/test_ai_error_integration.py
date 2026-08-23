from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from exceptions import LLMError


client = TestClient(app)


print("=" * 60)
print("TEST 1: LLM Error on Generate")


with patch(
    "router.ai.ai_service.llm_client.generate",
    side_effect=LLMError("LLM provider failed"),
):

    response = client.post(
        "/ai/generate",
        json={
            "message": "Docker چیست؟"
        },
    )


assert response.status_code == 503

assert response.json() == {
    "detail": "AI service is temporarily unavailable"
}

print("PASS")


print("=" * 60)
print("TEST 2: LLM Error on Chat")


with patch(
    "router.ai.ai_service.llm_client.chat",
    side_effect=LLMError("LLM provider failed"),
):

    response = client.post(
        "/ai/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Docker چیست؟"
                }
            ]
        },
    )


assert response.status_code == 503

assert response.json() == {
    "detail": "AI service is temporarily unavailable"
}

print("PASS")


print("=" * 60)
print("TEST 3: Normal Request Still Works")


with patch(
    "router.ai.ai_service.llm_client.generate",
    return_value="Docker یک ابزار کانتینری‌سازی است.",
):

    response = client.post(
        "/ai/generate",
        json={
            "message": "Docker چیست؟"
        },
    )


assert response.status_code == 200

assert response.json() == {
    "answer": "Docker یک ابزار کانتینری‌سازی است."
}

print("PASS")


print("=" * 60)
print("ALL AI ERROR INTEGRATION TESTS PASSED")