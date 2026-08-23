from pydantic import ValidationError

from schemas import (
    AIGenerateRequest,
    AIChatRequest,
    AIMessage,
    AIRoute,
    UserCreate,
)


print("=" * 60)
print("TEST 1: Valid AI Generate Request")

request = AIGenerateRequest(
    message="Docker چیست؟"
)

print(request)
print("PASS")


print("=" * 60)
print("TEST 2: Empty AI Generate Request")

try:
    AIGenerateRequest(
        message="   "
    )

    print("FAIL")

except ValidationError as e:
    print("PASS")
    print(e)


print("=" * 60)
print("TEST 3: Valid AI Message")

message = AIMessage(
    role="user",
    content="Docker چیست؟"
)

print(message)
print("PASS")


print("=" * 60)
print("TEST 4: Invalid AI Message Role")

try:
    AIMessage(
        role="admin",
        content="Hello"
    )

    print("FAIL")

except ValidationError as e:
    print("PASS")
    print(e)


print("=" * 60)
print("TEST 5: Valid Chat Request")

chat = AIChatRequest(
    messages=[
        AIMessage(
            role="user",
            content="Docker چیست؟"
        ),
        AIMessage(
            role="assistant",
            content="Docker یک ابزار containerization است."
        ),
    ]
)

print(chat)
print("PASS")


print("=" * 60)
print("TEST 6: Empty Chat")

try:
    AIChatRequest(
        messages=[]
    )

    print("FAIL")

except ValidationError as e:
    print("PASS")
    print(e)


print("=" * 60)
print("TEST 7: Valid Route")

route = AIRoute(
    intent="technical",
    topic="Docker",
    level="beginner",
    language="Persian",
)

print(route)
print("PASS")


print("=" * 60)
print("TEST 8: Invalid Route Intent")

try:
    AIRoute(
        intent="random",
        topic="Docker",
    )

    print("FAIL")

except ValidationError as e:
    print("PASS")
    print(e)


print("=" * 60)
print("TEST 9: Invalid User")

try:
    UserCreate(
        username="ab@cd",
        password="123456"
    )

    print("FAIL")

except ValidationError as e:
    print("PASS")
    print(e)