"""Cut every normalized transcript into chunks with one of two methods: normalized -> chunks.
Run: python scripts/chunk_corpus.py --method window60-v1   (or chapter60-v1)"""
import argparse
import hashlib
import json
from statistics import mean

from pymongo import MongoClient

from draft_coach.chunks import CHAPTER_METHOD, METHOD, make_chapter_windows, make_windows
from draft_coach.config import MONGO_URI

parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=[METHOD, CHAPTER_METHOD], default=METHOD)
method = parser.parse_args().method

db = MongoClient(MONGO_URI)["draft_coach"]

for normalized in db["normalized"].find():
    source_id = normalized["_id"]
    source = db["sources"].find_one({"_id": source_id}, {"url": 1, "chapters": 1})
    chapters = source.get("chapters", []) if method == CHAPTER_METHOD else []
    # Fingerprint of the input: segments, plus chapters for the chapter method (a fixed title rebuilds the chunks)
    hashed = [normalized["segments"], chapters] if chapters else normalized["segments"]
    input_hash = hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest()
    existing = db["chunks"].find_one({"source_id": source_id, "method": method}, {"input_hash": 1})
    if existing and existing["input_hash"] == input_hash:
        print(f"{source_id:<35} unchanged")
        continue
    db["chunks"].delete_many({"source_id": source_id, "method": method})  # changed or new: rebuild this source
    segments = sorted(normalized["segments"], key=lambda s: s["start_s"])  # screen lists are appended
    # sources without chapters get the plain windows with both methods
    windows = make_chapter_windows(segments, chapters) if chapters else make_windows(segments)
    chunks = [{
        "_id": f"{source_id}:{method}:{window['start_s']}",  # method in the id: two versions can coexist
        "source_id": source_id,
        "method": method,
        "input_hash": input_hash,
        **window,
        "url": f"{source['url']}&t={window['start_s']}s",  # link to the exact minute of the video
    } for window in windows]
    db["chunks"].insert_many(chunks)
    words = mean(len(chunk["text"].split()) for chunk in chunks)
    print(f"{source_id:<35} {len(chunks):>4} chunks {words:>6.0f} words on average")

# End-of-step check: every normalized transcript has chunks; statistics of every method side by side
print(f"\n{'method':<15} {'sources':>8} {'chunks':>7} {'words min / median / max':>26} {'under 50':>9}")
for name in sorted(db["chunks"].distinct("method")):
    words = sorted(len(c["text"].split()) for c in db["chunks"].find({"method": name}, {"text": 1}))
    n_sources = len(db["chunks"].distinct("source_id", {"method": name}))
    print(f"{name:<15} {n_sources:>8} {len(words):>7} "
          f"{f'{words[0]} / {words[len(words) // 2]} / {words[-1]}':>26} {sum(w < 50 for w in words):>9}")
assert db["normalized"].count_documents({}) == len(db["chunks"].distinct("source_id", {"method": method})), \
    "a normalized transcript has no chunks"