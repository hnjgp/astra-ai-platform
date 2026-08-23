from unittest.mock import Mock, patch
from schemas import AIRoute, AIMessage
from services.ai_service import AIService



def create_service():
    mock_llm = Mock()

    mock_llm.generate.return_value = "TEST ANSWER"

    return AIService(mock_llm), mock_llm


print("=" * 60)
print("TEST 1: Technical Request")

service, mock_llm = create_service()

technical_route = AIRoute(
    intent="technical",
    topic="Docker",
    level="beginner",
    language="Persian",
)

with patch(
    "services.ai_service.route_message",
    return_value=technical_route,
):

    result = service.generate("Docker چیست؟")

    assert result == "TEST ANSWER"

    mock_llm.generate.assert_called_once()

    call_args = mock_llm.generate.call_args

    assert call_args.kwargs["message"] == "Docker چیست؟"

    print("PASS")


print("=" * 60)
print("TEST 2: Career Request")

service, mock_llm = create_service()

career_route = AIRoute(
    intent="career",
    topic=None,
    level=None,
    language="Persian",
)

with patch(
    "services.ai_service.route_message",
    return_value=career_route,
):

    result = service.generate(
        "برای مصاحبه AI Engineer چطور آماده شوم؟"
    )

    assert result == "TEST ANSWER"

    mock_llm.generate.assert_called_once()

    call_args = mock_llm.generate.call_args

    assert (
        call_args.kwargs["message"]
        == "برای مصاحبه AI Engineer چطور آماده شوم؟"
    )

    print("PASS")


print("=" * 60)
print("TEST 3: General Request")

service, mock_llm = create_service()

general_route = AIRoute(
    intent="general",
    topic=None,
    level=None,
    language="Persian",
)

with patch(
    "services.ai_service.route_message",
    return_value=general_route,
):

    result = service.generate(
        "سلام، حالت چطوره؟"
    )

    assert result == "TEST ANSWER"

    mock_llm.generate.assert_called_once()

    print("PASS")


print("=" * 60)
print("TEST 4: Router Error")

service, mock_llm = create_service()

with patch(
    "services.ai_service.route_message",
    side_effect=Exception("Router failed"),
):

    result = service.generate("Docker چیست؟")

    assert (
        result
        == "متأسفانه در پردازش درخواست شما مشکلی پیش آمد."
    )

    print("PASS")


print("=" * 60)
print("TEST 5: LLM Error")

service, mock_llm = create_service()

mock_llm.generate.side_effect = Exception(
    "LLM failed"
)

technical_route = AIRoute(
    intent="technical",
    topic="Docker",
    level="beginner",
    language="Persian",
)

with patch(
    "services.ai_service.route_message",
    return_value=technical_route,
):

    result = service.generate("Docker چیست؟")

    assert (
        result
        == "متأسفانه در پردازش درخواست شما مشکلی پیش آمد."
    )

    print("PASS")


print("=" * 60)
print("ALL AI SERVICE UNIT TESTS PASSED")
print("=" * 60)
print("TEST 6: AIService Chat")

service, mock_llm = create_service()

messages = [
    AIMessage(
        role="user",
        content="Docker چیست؟",
    ),
    AIMessage(
        role="assistant",
        content="Docker یک ابزار کانتینری‌سازی است.",
    ),
    AIMessage(
        role="user",
        content="معماری آن را توضیح بده.",
    ),
]

mock_llm.chat.return_value = "TEST CHAT ANSWER"

result = service.chat(messages)

assert result == "TEST CHAT ANSWER"

mock_llm.chat.assert_called_once_with(messages)

print("PASS")
print("=" * 60)
print("TEST 7: Chat Error")

service, mock_llm = create_service()

mock_llm.chat.side_effect = Exception(
    "Chat failed"
)

messages = [
    AIMessage(
        role="user",
        content="Docker چیست؟",
    ),
]

try:

    service.chat(messages)

    assert False, "Expected exception was not raised"

except Exception as e:

    assert str(e) == "Chat failed"

    print("PASS")