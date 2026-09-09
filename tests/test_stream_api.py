# tests/test_stream_api.py

import time

import pytest
import requests


URL = "http://127.0.0.1:8000/ai/chat/stream"

PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "در سه پاراگراف کوتاه Docker را توضیح بده.",
        }
    ]
}


def test_stream_api():

    try:
        requests.get(
            "http://127.0.0.1:8000/health",
            timeout=2,
        )
    except requests.RequestException:
        pytest.skip(
            "API server is not running on 127.0.0.1:8000"
        )

    start_time = time.perf_counter()

    first_chunk_time = None
    chunk_count = 0

    with requests.post(
        URL,
        json=PAYLOAD,
        stream=True,
        timeout=60,
    ) as response:

        assert response.status_code == 200

        for chunk in response.iter_content(
            chunk_size=None,
            decode_unicode=True,
        ):

            if chunk:

                chunk_count += 1

                if first_chunk_time is None:
                    first_chunk_time = (
                        time.perf_counter()
                    )

                print(
                    chunk,
                    end="",
                    flush=True,
                )

    end_time = time.perf_counter()

    total_time = (
        end_time
        - start_time
    )

    assert chunk_count > 0
    assert first_chunk_time is not None

    ttft = (
        first_chunk_time
        - start_time
    )

    print()
    print("-" * 40)
    print("STREAM END")
    print(
        f"CHUNKS: {chunk_count}"
    )
    print(
        f"TOTAL TIME: {total_time:.3f}s"
    )
    print(
        f"TTFT: {ttft:.3f}s"
    )