from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


print("=" * 60)
print("TEST 1: Technical Integration")


with patch(
    "router.ai.ai_service.llm_client.generate",
    return_value="Docker یک ابزار برای اجرای کانتینرهاست.",
):

    response = client.post(
        "/ai/generate",
        json={
            "message": "Docker چیست؟"
        },
    )


assert response.status_code == 200

data = response.json()

assert "answer" in data
assert data["answer"] == "Docker یک ابزار برای اجرای کانتینرهاست."

print("PASS")


print("=" * 60)
print("TEST 2: Career Integration")


with patch(
    "router.ai.ai_service.llm_client.generate",
    return_value="برای مصاحبه AI Engineer باید روی Python، ML، LLM و System Design تمرکز کنید.",
):

    response = client.post(
        "/ai/generate",
        json={
            "message": "برای مصاحبه AI Engineer چطور آماده شوم؟"
        },
    )


assert response.status_code == 200

data = response.json()

assert "answer" in data

print("PASS")


print("=" * 60)
print("TEST 3: General Integration")


with patch(
    "router.ai.ai_service.llm_client.generate",
    return_value="سلام! ممنون، خوبم.",
):

    response = client.post(
        "/ai/generate",
        json={
            "message": "سلام، حالت چطوره؟"
        },
    )


assert response.status_code == 200

data = response.json()

assert data["answer"] == "سلام! ممنون، خوبم."

print("PASS")


print("=" * 60)
print("TEST 4: Chat Integration")


with patch(
    "router.ai.ai_service.llm_client.chat",
    return_value="Docker یک ابزار containerization است.",
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


assert response.status_code == 200

data = response.json()

assert data["answer"] == "Docker یک ابزار containerization است."

print("PASS")


print("=" * 60)
print("ALL AI INTEGRATION TESTS PASSED")