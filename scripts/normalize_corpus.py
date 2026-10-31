"""Normalize brawler names in every transcript: sources -> normalized, ambiguous names -> name_candidates.
Point corrections (data/corrections.yaml) are applied first, on the raw text, then the safe aliases.
Run: python scripts/normalize_corpus.py"""
from collections import Counter
from pathlib import Path

import yaml
from pymongo import MongoClient

from draft_coach.config import MONGO_URI
from draft_coach.names import NameNormalizer, apply_corrections, load_brawlers

normalizer = NameNormalizer(load_brawlers(Path("data/brawlers.yaml")))
db = MongoClient(MONGO_URI)["draft_coach"]
corrections = yaml.safe_load(Path("data/corrections.yaml").read_text(encoding="utf-8"))
unknown = {c["source"] for c in corrections} - set(db["sources"].distinct("_id", {"type": "transcript"}))
assert not unknown, f"corrections for sources that are not transcripts: {unknown}"  # would be skipped silently
replaced_total = Counter()
candidates = {}  # id "<source>:<segment>:<alias>" -> candidate (same alias twice in a segment = one candidate)

for source in db["sources"].find({"type": "transcript"}):
    segments = []
    for i, segment in enumerate(apply_corrections(source["_id"], source["segments"], corrections)):
        text, replaced = normalizer.normalize(segment["text"])
        replaced_total.update(replaced)
        segments.append({"start_s": segment["start_s"], "text": text, "mentions": normalizer.mentions(text)})
        for alias in normalizer.context_candidates(text):
            candidates[f"{source['_id']}:{i}:{alias}"] = {
                "source_id": source["_id"], "segment": i, "start_s": segment["start_s"], "alias": alias,
                "brawler_ids": normalizer.context_to_ids[alias], "text": text}
    db["normalized"].replace_one({"_id": source["_id"]}, {"segments": segments}, upsert=True)
    n_mentions = sum(len(s["mentions"]) for s in segments)
    print(f"{source['_id']:<35} {len(segments):>5} segments {n_mentions:>5} mentions")

# Rebuild the collection on every run (idempotent); the id stays the same, so step 2c proposals still match
db["name_candidates"].delete_many({})
db["name_candidates"].insert_many([{"_id": key, **candidate} for key, candidate in candidates.items()])

# End-of-step check
print(f"\n{'point corrections applied':<35} {len(corrections)}")  # each one matched exactly once, or the run stopped
print(f"{'aliases replaced':<35} {sum(replaced_total.values())}")
for alias, count in replaced_total.most_common(10):
    print(f"  {alias:<33} {count}")
print(f"{'context candidates (step 2c)':<35} {len(candidates)}")
for alias, count in Counter(c["alias"] for c in candidates.values()).most_common(10):
    print(f"  {alias:<33} {count}")
n_transcripts = db["sources"].count_documents({"type": "transcript"})
print(f"{'transcripts / normalized':<35} {n_transcripts} / {db['normalized'].count_documents({})}")
assert n_transcripts == db["normalized"].count_documents({}), "a transcript was not normalized"