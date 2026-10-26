"""Cut every normalized transcript into time windows: normalized -> chunks.
Run: python scripts/chunk_corpus.py"""
import hashlib
import json
from statistics import mean

from pymongo import MongoClient

from draft_coach.chunks import METHOD, make_windows
from draft_coach.config import MONGO_URI

db = MongoClient(MONGO_URI)["draft_coach"]

for normalized in db["normalized"].find():
    source_id = normalized["_id"]
    # Fingerprint of the input: if the normalized text changes (e.g. new name decisions), the hash changes
    input_hash = hashlib.sha256(json.dumps(normalized["segments"], sort_keys=True).encode()).hexdigest()
    existing = db["chunks"].find_one({"source_id": source_id, "method": METHOD}, {"input_hash": 1})
    if existing and existing["input_hash"] == input_hash:
        print(f"{source_id:<35} unchanged")
        continue
    db["chunks"].delete_many({"source_id": source_id, "method": METHOD})  # changed or new: rebuild this source
    url = db["sources"].find_one({"_id": source_id}, {"url": 1})["url"]
    chunks = [{
        "_id": f"{source_id}:{METHOD}:{window['start_s']}",  # method in the id: two versions can coexist
        "source_id": source_id,
        "method": METHOD,
        "input_hash": input_hash,
        **window,
        "url": f"{url}&t={window['start_s']}s",  # link to the exact minute of the video
    } for window in make_windows(sorted(normalized["segments"], key=lambda s: s["start_s"]))]  # screen lists are appended
    db["chunks"].insert_many(chunks)
    words = mean(len(chunk["text"].split()) for chunk in chunks)
    print(f"{source_id:<35} {len(chunks):>4} chunks {words:>6.0f} words on average")

# End-of-step check: every normalized transcript has chunks; one chunk printed in full
chunked = db["chunks"].distinct("source_id", {"method": METHOD})
all_chunks = list(db["chunks"].find({"method": METHOD}, {"text": 1}))
print(f"\n{'transcripts / chunked':<35} {db['normalized'].count_documents({})} / {len(chunked)}")
print(f"{'chunks':<35} {len(all_chunks)}")
print(f"{'words per chunk (min / mean / max)':<35} "
      f"{min(len(c['text'].split()) for c in all_chunks)} / "
      f"{mean(len(c['text'].split()) for c in all_chunks):.0f} / "
      f"{max(len(c['text'].split()) for c in all_chunks)}")
sample = db["chunks"].find_one({"method": METHOD}, sort=[("_id", 1)], skip=len(all_chunks) // 2)
print(f"\nSample {sample['_id']}  ({sample['start_s']}-{sample['end_s']} s)\n{sample['url']}\n"
      f"mentions: {sample['mentions']}\n{sample['text']}")
assert db["normalized"].count_documents({}) == len(chunked), "a normalized transcript has no chunks"