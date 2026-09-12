from rag.embeddings.embedder import OpenAIEmbedder


class FakeEmbeddings:
    def create(self, model, input):
        assert model == "text-embedding-3-small"
        assert input == "hello world"

        return type(
            "Response",
            (),
            {
                "data": [
                    type(
                        "EmbeddingData",
                        (),
                        {
                            "embedding": [0.1, 0.2, 0.3]
                        },
                    )()
                ]
            },
        )()


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


def test_openai_embedder_returns_embedding():
    embedder = OpenAIEmbedder(
        client=FakeClient()
    )

    result = embedder.embed("hello world")

    assert result == [0.1, 0.2, 0.3]