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

class FakeBatchEmbeddings:
    def __init__(self):
        self.requests = []

    def create(self, model, input):
        self.requests.append(list(input))

        # Return items out of order to check that the index is used.
        items = [
            type(
                "EmbeddingData",
                (),
                {
                    "index": index,
                    "embedding": [float(len(text))],
                },
            )()
            for index, text in enumerate(input)
        ]

        return type(
            "Response",
            (),
            {"data": list(reversed(items)), "usage": None},
        )()


def test_embed_many_batches_requests_and_keeps_order():
    client = FakeClient()
    client.embeddings = FakeBatchEmbeddings()

    embedder = OpenAIEmbedder(client=client)
    embedder.batch_size = 2

    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    result = embedder.embed_many(texts)

    assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    # 5 texts with batch size 2 -> 3 requests instead of 5.
    assert client.embeddings.requests == [
        ["a", "bb"],
        ["ccc", "dddd"],
        ["eeeee"],
    ]
