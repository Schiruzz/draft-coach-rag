"""Normalize brawler names in every transcript: sources -> normalized, ambiguous names -> name_candidates.
Run: python scripts/normalize_corpus.py"""
from collections import Counter
from pathlib import Path

from pymongo import MongoClient

from draft_coach.config import MONGO_URI
from draft_coach.names import NameNormalizer, load_brawlers

normalizer = NameNormalizer(load_brawlers(Path("data/brawlers.yaml")))
db = MongoClient(MONGO_URI)["draft_coach"]
replaced_total = Counter()
candidates = []

for source in db["sources"].find({"type": "transcript"}):
    segments = []
    for segment in source["segments"]:
        text, replaced = normalizer.normalize(segment["text"])
        replaced_total.update(replaced)
        segments.append({"start_s": segment["start_s"], "text": text, "mentions": normalizer.mentions(text)})
        for alias in normalizer.context_candidates(text):
            candidates.append({"source_id": source["_id"], "start_s": segment["start_s"], "alias": alias,
                               "brawler_id": normalizer.context_to_id[alias], "text": text})
    db["normalized"].replace_one({"_id": source["_id"]}, {"segments": segments}, upsert=True)
    n_mentions = sum(len(s["mentions"]) for s in segments)
    print(f"{source['_id']:<35} {len(segments):>5} segments {n_mentions:>5} mentions")

# Candidates have no natural id: rebuild the collection on every run (idempotent)
db["name_candidates"].delete_many({})
db["name_candidates"].insert_many(candidates)

# End-of-step check
print(f"\n{'aliases replaced':<35} {sum(replaced_total.values())}")
for alias, count in replaced_total.most_common(10):
    print(f"  {alias:<33} {count}")
print(f"{'context candidates (step 2c)':<35} {len(candidates)}")
for alias, count in Counter(c["alias"] for c in candidates).most_common(10):
    print(f"  {alias:<33} {count}")
n_transcripts = db["sources"].count_documents({"type": "transcript"})
print(f"{'transcripts / normalized':<35} {n_transcripts} / {db['normalized'].count_documents({})}")
assert n_transcripts == db["normalized"].count_documents({}), "a transcript was not normalized"