from prompts.router import route_message


messages = [
    "Docker چیست؟",
    "Docker را توضیح بده",
    "Explain PostgreSQL",
    "سلام، حالت چطوره؟",
    "امروز هوا چطوره؟",
]


for message in messages:
    result = route_message(message)

    print("=" * 50)
    print("MESSAGE:", message)
    print("INTENT:", result.intent)
    print("TOPIC:", result.topic)