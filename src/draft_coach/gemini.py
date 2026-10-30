"""One structured Gemini call with retries on temporary errors (used by the claim extraction)."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

import httpx
from google import genai
from google.genai import errors, types

ATTEMPTS = 5
RETRY_CODES = {408, 429, 499, 500, 502, 503, 504}  # temporary: queue on Vertex, quota, server errors


def generate_json(client: genai.Client, model: str, prompt: str, schema, thinking_level: str, label: str = ""):
    """Parsed answer of `schema`, up to ATTEMPTS calls; None if it keeps failing (asked again on the next run).
    The client carries the timeout; every failed attempt prints one line."""
    for attempt in range(1, ATTEMPTS + 1):
        start = time.monotonic()
        try:
            response = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_schema=schema, temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level)))
            return response.parsed
        except (errors.APIError, httpx.TransportError) as error:  # TransportError: network timeout, dropped connection
            if isinstance(error, errors.APIError) and error.code not in RETRY_CODES:
                raise
            reason = f"{error.code} {error.status}" if isinstance(error, errors.APIError) else type(error).__name__
            print(f"{label}: {reason} after {time.monotonic() - start:.0f} s, attempt {attempt}/{ATTEMPTS}")
            if attempt < ATTEMPTS:
                time.sleep(2 ** attempt)  # 2, 4, 8, 16 s
    return None


def generate_json_hedged(client: genai.Client, model: str, prompt: str, schema, thinking_level: str,
                         hedge_after_s: float, label: str = "") -> tuple[object | None, bool, bool]:
    """generate_json with a backup request: if no answer after hedge_after_s, an identical copy is sent and the
    first valid answer wins (slow calls are queueing on the server, not working). Returns (answer, backup sent,
    backup won). The losing call cannot be stopped mid-flight: it finishes in the background and is ignored."""
    pool = ThreadPoolExecutor(2)
    first = pool.submit(generate_json, client, model, prompt, schema, thinking_level, label)
    done, _ = wait([first], timeout=hedge_after_s)
    futures = {first: False}
    if not done:
        futures[pool.submit(generate_json, client, model, prompt, schema, thinking_level, f"{label} (backup)")] = True
    try:
        for future in as_completed(futures):
            answer = future.result()
            if answer is not None:
                return answer, len(futures) == 2, futures[future]
        return None, len(futures) == 2, False
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # do not wait for the losing call