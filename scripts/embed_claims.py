"""Step 6: embed every claim with one text variant and store it in Qdrant with its payload: claims (MongoDB) -> vectors.
Each point carries its parent chunk (window60-v1): search the claim, read the chunk (small-to-big).
Run: python scripts/embed_claims.py --text concept   (or concept_reason)"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import errors, types
from pymongo import MongoClient
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from draft_coach.chunks import METHOD
from draft_coach.claims import EXTRACT_VERSION, claim_text, parent_chunk
from draft_coach.config import EMBEDDING_MODEL, MONGO_URI, QDRANT_URL
from draft_coach.vectors import point_id, text_hash

DIMENSION = 3072     # gemini-embedding-2, as for the chunks
PARALLEL_CALLS = 4   # as embed_chunks.py: one vector per call
RETRY_CODES = {408, 429, 499, 500, 502, 503, 504}
UPSERT_BATCH = 64
PAYLOAD = ("source_id", "url", "start_s", "end_s", "concept", "reason", "example", "topic", "relation",
           "brawlers", "classes", "effects", "stats", "against_brawlers", "against_classes", "against_effects",
           "brawler_classes", "against_brawler_classes", "mentions", "draft_phase", "rank_band", "modes", "map_ids",
           "map_types", "map_features", "lane_role", "mode_role", "mode_role_raw")  # fields for the filters of step 7b

parser = argparse.ArgumentParser()
parser.add_argument("--text", choices=["concept", "concept_reason"], default="concept_reason")
variant = parser.parse_args().text

db = MongoClient(MONGO_URI)["draft_coach"]
claims = list(db["claims"].find())
chunks_by_source = {}  # transcript chunks of the chosen method, per source (the wiki layer has no time)
for c in db["chunks"].find({"method": METHOD, "layer": {"$exists": False}}, {"source_id": 1, "start_s": 1, "end_s": 1}):
    chunks_by_source.setdefault(c["source_id"], []).append(c)

qdrant = QdrantClient(url=QDRANT_URL)
collection = f"claims__{EXTRACT_VERSION}__{variant}__{EMBEDDING_MODEL}"  # vectors of different texts never mix
if not qdrant.collection_exists(collection):
    qdrant.create_collection(collection, vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE))

# Incremental: compare the text hash stored in Qdrant with the current claim text
stored, offset = {}, None
while True:
    points, offset = qdrant.scroll(collection, limit=1000, offset=offset, with_payload=["text_hash"])
    stored.update({str(p.id): p.payload["text_hash"] for p in points})
    if offset is None:
        break
current = {point_id(c["_id"]): c for c in claims}
todo = [c for pid, c in current.items() if stored.get(pid) != text_hash(claim_text(c, variant))]
stale = [pid for pid in stored if pid not in current]  # claims of sections extracted again
if stale:
    qdrant.delete(collection, points_selector=models.PointIdsList(points=stale))
print(f"claims {len(claims)}   already embedded {len(claims) - len(todo)}   to embed {len(todo)}   stale removed {len(stale)}")

genai_client = genai.Client(http_options=types.HttpOptions(timeout=30_000))  # ms; project and location from .env
config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")            # claims are documents


def embed(claim: dict) -> list[float] | None:
    """One claim -> one vector; retries on temporary errors, None if it keeps failing (taken next run)."""
    for attempt in range(5):
        try:
            response = genai_client.models.embed_content(model=EMBEDDING_MODEL, contents=claim_text(claim, variant),
                                                         config=config)
            return response.embeddings[0].values
        except errors.APIError as error:
            if error.code not in RETRY_CODES:
                raise
            time.sleep(2 ** attempt)
    return None


points, failed, orphans = [], 0, 0
with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
    for claim, vector in tqdm(zip(todo, pool.map(embed, todo)), total=len(todo), desc="embedding"):
        if vector is None:
            failed += 1
            continue
        parent = parent_chunk(claim, chunks_by_source.get(claim["source_id"], []))
        orphans += parent is None
        text = claim_text(claim, variant)
        payload = {"claim_id": claim["_id"], "chunk_id": parent, "text": text, "text_hash": text_hash(text),
                   **{key: claim[key] for key in PAYLOAD}}
        points.append(models.PointStruct(id=point_id(claim["_id"]), vector=vector, payload=payload))
        if len(points) == UPSERT_BATCH:
            qdrant.upsert(collection, points)
            points = []
if points:
    qdrant.upsert(collection, points)

# End-of-step check: one point per claim, every claim with its chunk
n_points = qdrant.count(collection).count
print(f"\n{'collection':<20} {collection}")
print(f"{'claims / points':<20} {len(claims)} / {n_points}")
print(f"{'failed (next run)':<20} {failed}")
print(f"{'without chunk':<20} {orphans}")
assert n_points == len(claims), "Qdrant and MongoDB disagree: rerun to embed the missing claims"