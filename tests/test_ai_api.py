from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


print("=" * 60)
print("TEST 1: Valid Generate Request")

with patch(
    "router.ai.ai_service.generate",
    return_value="Docker یک پلتفرم کانتینری‌سازی است.",
):
    response = client.post(
        "/ai/generate",
        json={
            "message": "Docker چیست؟"
        },
    )

assert response.status_code == 200
assert response.json() == {
    "answer": "Docker یک پلتفرم کانتینری‌سازی است."
}

print("PASS")


print("=" * 60)
print("TEST 2: Empty Generate Request")

response = client.post(
    "/ai/generate",
    json={
        "message": "   "
    },
)

assert response.status_code == 422

print("PASS")


print("=" * 60)
print("TEST 3: Valid Chat Request")

with patch(
    "router.ai.ai_service.chat",
    return_value="Docker یک پلتفرم کانتینری‌سازی است.",
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
assert response.json() == {
    "answer": "Docker یک پلتفرم کانتینری‌سازی است."
}

print("PASS")


print("=" * 60)
print("TEST 4: Empty Chat")

response = client.post(
    "/ai/chat",
    json={
        "messages": []
    },
)

assert response.status_code == 422

print("PASS")


print("=" * 60)
print("TEST 5: Invalid Chat Role")

response = client.post(
    "/ai/chat",
    json={
        "messages": [
            {
                "role": "admin",
                "content": "Docker چیست؟"
            }
        ]
    },
)

assert response.status_code == 422

print("PASS")


print("=" * 60)
print("ALL AI API TESTS PASSED")