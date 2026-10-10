class FakeLLMClient:

    def __init__(self):
        self.calls = []

    def generate(self, message: str, instructions: str) -> str:

        self.calls.append(
            {
                "message": message,
                "instructions": instructions,
            }
        )

        return "FAKE LLM RESPONSE"

    def chat(self, messages) -> str:
        self.calls.append(
            {
                "messages": messages,
            }
        )

        return "FAKE CHAT RESPONSE"

class ScriptedLLMClient:
    """
    Returns pre-defined Responses-API-like objects, one per call.

    Each script item is either a final answer (str) or a list of
    (tool_name, arguments_json) tool calls.
    """

    def __init__(self, script):
        from types import SimpleNamespace

        self._namespace = SimpleNamespace
        self.script = list(script)
        self.calls = []

    def generate(self, message, instructions):
        self.calls.append(
            {
                "message": message,
                "instructions": instructions,
            }
        )

        return self.script.pop(0)

    def generate_with_tools(
        self,
        message,
        tools,
        instructions=None,
        previous_response_id=None,
    ):
        self.calls.append(
            {
                "message": message,
                "tools": tools,
                "instructions": instructions,
                "previous_response_id": previous_response_id,
            }
        )

        step = self.script.pop(0)
        response_id = f"response_{len(self.calls)}"

        if isinstance(step, str):
            return self._namespace(
                id=response_id,
                output=[],
                output_text=step,
            )

        return self._namespace(
            id=response_id,
            output=[
                self._namespace(
                    type="function_call",
                    call_id=f"call_{len(self.calls)}_{index}",
                    name=name,
                    arguments=arguments,
                )
                for index, (name, arguments) in enumerate(step)
            ],
            output_text="",
        )


class FakeEmbedder:
    """
    Deterministic bag-of-words embedder with 1536 dimensions.

    Texts that share words get similar vectors, which is enough to
    test retrieval without calling an embedding provider.
    """

    dimensions = 1536

    def __init__(self):
        self.calls = 0
        self.batch_calls = 0

    def embed(self, text):
        self.calls += 1
        return self._vector(text)

    def embed_many(self, texts):
        self.batch_calls += 1
        return [self._vector(text) for text in texts]

    def _vector(self, text):
        import hashlib
        import math
        import re

        vector = [0.0] * self.dimensions

        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            vector[0] = 1.0
            return vector

        return [value / norm for value in vector]
