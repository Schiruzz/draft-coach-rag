"""Probe the embedding models on Vertex AI: which ones answer, how fast, how many numbers per vector,
and whether one call accepts a batch of chunks. Run: python scripts/probe_embedding.py"""
import time

from google import genai
from google.genai import types
from pymongo import MongoClient

from draft_coach.chunks import METHOD
from draft_coach.config import MONGO_URI

# Candidate model ids, to be confirmed by this probe (Vertex names change often)
CANDIDATES = ["gemini-embedding-2", "gemini-embedding-001", "text-multilingual-embedding-002"]
BATCH_TEST = 20  # chunks sent in a single call in the second test

texts = [c["text"] for c in MongoClient(MONGO_URI)["draft_coach"]["chunks"].find({"method": METHOD}).limit(BATCH_TEST)]
client = genai.Client(http_options=types.HttpOptions(timeout=30_000))  # ms; project and location from .env
config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")      # chunks are documents, questions are queries

for model in CANDIDATES:
    for label, contents in (("1 chunk", texts[:1]), (f"{BATCH_TEST} chunks", texts)):
        try:
            start = time.perf_counter()
            response = client.models.embed_content(model=model, contents=contents, config=config)
            seconds = time.perf_counter() - start
            print(f"{model:<34} {label:<10} ok  {seconds:5.1f} s  "
                  f"{len(response.embeddings)} vectors x {len(response.embeddings[0].values)} numbers")
        except Exception as error:  # a probe: show the error and go on with the next case
            print(f"{model:<34} {label:<10} ERROR {str(error)[:120]}")