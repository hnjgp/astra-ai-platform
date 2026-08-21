from schemas import AIRoute


def route_message(message: str) -> AIRoute:
    normalized = message.strip().lower()

    teaching_patterns = [
        "چیست",
        "چیه",
        "چی میشه",
        "توضیح بده",
        "توضیحش بده",
        "معنی",
        "what is",
        "explain",
        "describe",
    ]

    is_teaching = any(
        pattern in normalized
        for pattern in teaching_patterns
    )

    if not is_teaching:
        return AIRoute(
            intent="general",
            topic=None,
        )

    topic = extract_topic(message)

    return AIRoute(
        intent="teaching",
        topic=topic,
    )


def extract_topic(message: str) -> str:
    text = message.strip()

    separators = [
        " چیست؟",
        " چیست",
        " چیه؟",
        " چیه",
        " را توضیح بده",
        " رو توضیح بده",
        " توضیح بده",
    ]

    for separator in separators:
        if separator in text:
            topic = text.split(separator, 1)[0].strip()

            if topic:
                return topic

    lower_text = text.lower()

    english_prefixes = [
        "explain ",
        "describe ",
        "what is ",
    ]

    for prefix in english_prefixes:
        if lower_text.startswith(prefix):
            return text[len(prefix):].strip().rstrip("؟?")

    return text