"""Step 6: embed every claim with one text variant and store it in Qdrant with its payload: claims (MongoDB) -> vectors.
Each point carries its parent chunk (of the method in use): search the claim, read the chunk (small-to-big).
The parent is checked on every run: a new chunking method relinks the points without embedding them again.
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
from draft_coach.claims import EXTRACT_VERSION, claim_text, parent_chunk, stale_links
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
chunks_by_source = {}  # transcript chunks of the method in use, per source (the wiki layer has no time)
for c in db["chunks"].find({"method": METHOD, "layer": {"$exists": False}}, {"source_id": 1, "start_s": 1, "end_s": 1}):
    chunks_by_source.setdefault(c["source_id"], []).append(c)
# The parent depends on the claim AND on the chunking method: computed for every claim on every run
parents = {point_id(c["_id"]): parent_chunk(c, chunks_by_source.get(c["source_id"], [])) for c in claims}

qdrant = QdrantClient(url=QDRANT_URL)
collection = f"claims__{EXTRACT_VERSION}__{variant}__{EMBEDDING_MODEL}"  # vectors of different texts never mix
if not qdrant.collection_exists(collection):
    qdrant.create_collection(collection, vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE))


def stored_payloads(fields: list[str]) -> dict[str, dict]:
    """Point id -> the given payload fields, for every point of the collection."""
    payloads, offset = {}, None
    while True:
        points, offset = qdrant.scroll(collection, limit=1000, offset=offset, with_payload=fields)
        payloads.update({str(p.id): p.payload for p in points})
        if offset is None:
            return payloads


# Incremental: compare the text hash stored in Qdrant with the current claim text
stored = stored_payloads(["text_hash", "chunk_id"])
current = {point_id(c["_id"]): c for c in claims}
todo = [c for pid, c in current.items() if stored.get(pid, {}).get("text_hash") != text_hash(claim_text(c, variant))]
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


points, failed = [], 0
with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
    for claim, vector in tqdm(zip(todo, pool.map(embed, todo)), total=len(todo), desc="embedding"):
        if vector is None:
            failed += 1
            continue
        text = claim_text(claim, variant)
        payload = {"claim_id": claim["_id"], "chunk_id": parents[point_id(claim["_id"])], "text": text,
                   "text_hash": text_hash(text), **{key: claim[key] for key in PAYLOAD}}
        points.append(models.PointStruct(id=point_id(claim["_id"]), vector=vector, payload=payload))
        if len(points) == UPSERT_BATCH:
            qdrant.upsert(collection, points)
            points = []
if points:
    qdrant.upsert(collection, points)

# Points embedded before whose parent chunk changed (a new chunking method): new payload, same vector
embedded = {point_id(c["_id"]) for c in todo}
old_links = {pid: payload.get("chunk_id") for pid, payload in stored.items() if pid in current and pid not in embedded}
relinked = stale_links(old_links, parents)
operations = [models.SetPayloadOperation(set_payload=models.SetPayload(payload={"chunk_id": chunk}, points=[pid]))
              for pid, chunk in relinked.items()]
for i in range(0, len(operations), UPSERT_BATCH):
    qdrant.batch_update_points(collection, update_operations=operations[i:i + UPSERT_BATCH])

# End-of-step check: one point per claim, every point linked to a chunk of the method in use
n_points = qdrant.count(collection).count
known_chunks = {c["_id"] for chunks in chunks_by_source.values() for c in chunks}
links = [payload.get("chunk_id") for payload in stored_payloads(["chunk_id"]).values()]
foreign = sum(link is not None and link not in known_chunks for link in links)
print(f"\n{'collection':<24} {collection}")
print(f"{'chunk method':<24} {METHOD}")
print(f"{'claims / points':<24} {len(claims)} / {n_points}")
print(f"{'failed (next run)':<24} {failed}")
print(f"{'relinked (no embedding)':<24} {len(relinked)}")
print(f"{'without chunk':<24} {sum(link is None for link in links)}")
print(f"{'linked to another method':<24} {foreign}")
assert n_points == len(claims), "Qdrant and MongoDB disagree: rerun to embed the missing claims"
assert foreign == 0, "some claims point to chunks of another method: rerun"