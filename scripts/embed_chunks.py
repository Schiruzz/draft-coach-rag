"""Embed every chunk of one method and store it in Qdrant with its payload: chunks (MongoDB) -> vectors (Qdrant).
Run: python scripts/embed_chunks.py --method window60-v1   (or chapter60-v1)"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import errors, types
from pymongo import MongoClient
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from draft_coach.chunks import CHAPTER_METHOD, METHOD
from draft_coach.config import EMBEDDING_MODEL, MONGO_URI, QDRANT_URL
from draft_coach.vectors import collection_name, point_id, text_hash

DIMENSION = 3072     # measured with probe_embedding.py for gemini-embedding-2
PARALLEL_CALLS = 4   # the model returns ONE vector per call (a list of texts is merged), so we parallelize calls
RETRY_CODES = {408, 429, 499, 500, 502, 503, 504}
UPSERT_BATCH = 64

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=[METHOD, CHAPTER_METHOD], default=METHOD)
method = parser.parse_args().method

chunks = list(MongoClient(MONGO_URI)["draft_coach"]["chunks"].find({"method": method}))
qdrant = QdrantClient(url=QDRANT_URL)

collection = collection_name(method, EMBEDDING_MODEL)  # one collection per method
if not qdrant.collection_exists(collection):
    qdrant.create_collection(collection, vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE))

# Incremental: compare the text hash stored in Qdrant with the current chunk text
stored = {}  # point id -> text hash
offset = None
while True:
    points, offset = qdrant.scroll(collection, limit=1000, offset=offset, with_payload=["text_hash"])
    stored.update({str(p.id): p.payload["text_hash"] for p in points})
    if offset is None:
        break
current = {point_id(c["_id"]): c for c in chunks}
todo = [c for pid, c in current.items() if stored.get(pid) != text_hash(c["text"])]
stale = [pid for pid in stored if pid not in current]  # chunks rebuilt with other windows: their old points go
if stale:
    qdrant.delete(collection, points_selector=models.PointIdsList(points=stale))
print(f"chunks {len(chunks)}   already embedded {len(chunks) - len(todo)}   to embed {len(todo)}   stale removed {len(stale)}")

genai_client = genai.Client(http_options=types.HttpOptions(timeout=30_000))  # ms; project and location from .env
config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")            # chunks are documents


def embed(chunk: dict) -> list[float] | None:
    """One chunk -> one vector; retries on temporary errors, None if it keeps failing (taken next run)."""
    for attempt in range(5):
        try:
            response = genai_client.models.embed_content(model=EMBEDDING_MODEL, contents=chunk["text"], config=config)
            return response.embeddings[0].values
        except errors.APIError as error:
            if error.code not in RETRY_CODES:
                raise
            time.sleep(2 ** attempt)
    return None


points, failed = [], 0
with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
    for chunk, vector in tqdm(zip(todo, pool.map(embed, todo)), total=len(todo), desc="embedding"):
        if vector is None:
            failed += 1
            continue
        points.append(models.PointStruct(id=point_id(chunk["_id"]), vector=vector, payload={
            "chunk_id": chunk["_id"], "source_id": chunk["source_id"], "start_s": chunk["start_s"],
            "end_s": chunk["end_s"], "url": chunk["url"], "mentions": chunk["mentions"],
            "text": chunk["text"], "text_hash": text_hash(chunk["text"])}))
        if len(points) == UPSERT_BATCH:
            qdrant.upsert(collection, points)
            points = []
if points:
    qdrant.upsert(collection, points)

# End-of-step check: one point per chunk; one point printed
n_points = qdrant.count(collection).count
print(f"\n{'collection':<20} {collection}")
print(f"{'chunks / points':<20} {len(chunks)} / {n_points}")
print(f"{'failed (next run)':<20} {failed}")
sample = qdrant.retrieve(collection, [point_id(chunks[0]["_id"])], with_vectors=True)[0]
print(f"\nSample point {sample.payload['chunk_id']}: {len(sample.vector)} numbers, first 5 {[round(x, 4) for x in sample.vector[:5]]}")
print(f"payload: { {k: v for k, v in sample.payload.items() if k != 'text'} }")
assert n_points == len(chunks), "Qdrant and MongoDB disagree: rerun to embed the missing chunks"