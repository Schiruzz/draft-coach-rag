"""Which Gemini models answer, where, and how fast: one tiny call per model and location.
Run: python scripts/probe_models.py"""
import os
import time

from google import genai
from google.genai import types

MODELS = ["gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.5-flash-lite"]
LOCATIONS = ["eu", "global"]
CALLS = 3  # a few calls each: one slow call is not a measure

for location in LOCATIONS:
    client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location=location,
                          http_options=types.HttpOptions(timeout=30_000))
    for model in MODELS:
        times = []
        for _ in range(CALLS):
            start = time.monotonic()
            try:
                client.models.generate_content(model=model, contents="Say OK")
                times.append(f"{time.monotonic() - start:.1f}")
            except Exception as error:  # probe only: show any failure and move on
                times.append(type(error).__name__ + (f" {error.code}" if hasattr(error, "code") else ""))
        print(f"{location:<7} {model:<24} {'  '.join(times)}")