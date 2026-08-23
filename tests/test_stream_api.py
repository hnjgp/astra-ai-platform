import time

import requests


url = "http://127.0.0.1:8000/ai/chat/stream"

payload = {
    "messages": [
        {
            "role": "user",
            "content": "در سه پاراگراف کوتاه Docker را توضیح بده.",
        }
    ]
}


print("STREAM START")
print("-" * 40)


start_time = time.perf_counter()

first_chunk_time = None
chunk_count = 0


with requests.post(
    url,
    json=payload,
    stream=True,
) as response:

    print("STATUS:", response.status_code)

    response.raise_for_status()

    for chunk in response.iter_content(
        chunk_size=None,
        decode_unicode=True,
    ):

        if chunk:

            chunk_count += 1

            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()

                ttft = (
                    first_chunk_time
                    - start_time
                )

                print(
                    f"\n\nFIRST CHUNK: {ttft:.3f}s"
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


print()
print("-" * 40)
print("STREAM END")

print(
    f"CHUNKS: {chunk_count}"
)

print(
    f"TOTAL TIME: {total_time:.3f}s"
)

if first_chunk_time is not None:

    print(
        f"TTFT: {first_chunk_time - start_time:.3f}s"
    )