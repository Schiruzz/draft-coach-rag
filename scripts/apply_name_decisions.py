"""Measure Gemini on the blind sample, then apply the resolved ambiguous names to the normalized transcripts.
Run after normalize_corpus.py and propose_names.py: python scripts/apply_name_decisions.py"""
from pathlib import Path

import yaml
from pymongo import MongoClient

from draft_coach.config import MONGO_URI
from draft_coach.names import load_brawlers, word_pattern

PROMPT_VERSION = "v3"  # the Gemini proposals that were reviewed
REVIEW_DIR = Path("data/name_review")

db = MongoClient(MONGO_URI)["draft_coach"]
en_name = {b["id"]: b["en"] for b in load_brawlers(Path("data/brawlers.yaml"))}
proposals = {p["_id"]: p for p in db["name_proposals"].find({"prompt_version": PROMPT_VERSION})}
decisions = {e["id"]: e["decision"] for e in yaml.safe_load((REVIEW_DIR / "decisions.yaml").read_text(encoding="utf-8"))}
labels = {e["id"]: e["label"] for e in yaml.safe_load((REVIEW_DIR / "sample.yaml").read_text(encoding="utf-8"))}
assert None not in decisions.values() and None not in labels.values(), "review files still have empty entries"

# 1. Gemini against Federico's labels on the blind sample ("none" = not a brawler).
# Only high-confidence answers are measured: they are the ones accepted without review.
high = [i for i in labels if i in proposals and proposals[i]["confidence"] == "high"]  # sample items no longer candidates (now safe aliases) are skipped
gemini = {i: proposals[i]["brawler_id"] or "none" for i in high}
correct = sum(gemini[i] == labels[i] for i in high)
true_positives = sum(gemini[i] == labels[i] != "none" for i in high)
said_brawler = sum(gemini[i] != "none" for i in high)
is_brawler = sum(labels[i] != "none" for i in high)
print(f"{'sample size':<30} {len(high)} high confidence ({len(labels) - len(high)} low or no longer candidates, excluded)")
print(f"{'accuracy':<30} {correct / len(high):.1%}")
print(f"{'precision':<30} {true_positives / said_brawler:.1%}  (Gemini says brawler X: how often right)")
print(f"{'recall':<30} {true_positives / is_brawler:.1%}  (real brawler mentions Gemini finds)")

# 2. Final answer per candidate: Federico's decision or label first, otherwise Gemini (high confidence only)
resolved, source_count = {}, {"decision": 0, "label": 0, "gemini": 0}
for candidate_id, proposal in proposals.items():
    if candidate_id in decisions:
        answer, source = decisions[candidate_id], "decision"
    elif candidate_id in labels:
        answer, source = labels[candidate_id], "label"
    elif proposal["confidence"] == "high":
        answer, source = proposal["brawler_id"] or "none", "gemini"
    else:
        continue  # low confidence without a decision: left as it is
    source_count[source] += 1
    if answer != "none":
        resolved[candidate_id] = answer

# 3. Apply: alias -> English name in the segment text, brawler id added to mentions (idempotent)
added = 0
for doc in db["normalized"].find():
    segments = doc["segments"]
    for candidate_id, brawler_id in resolved.items():
        source_id, segment, alias = candidate_id.split(":", 2)
        if source_id != doc["_id"]:
            continue
        seg = segments[int(segment)]
        # every occurrence of the alias in this segment: candidates are one per (segment, alias)
        seg["text"] = word_pattern([alias]).sub(en_name[brawler_id], seg["text"])
        if brawler_id not in seg["mentions"]:
            seg["mentions"].append(brawler_id)
            added += 1
    db["normalized"].update_one({"_id": doc["_id"]}, {"$set": {"segments": segments}})

# End-of-step check
print(f"\n{'answers used':<30} " + ", ".join(f"{k} {v}" for k, v in source_count.items()))
print(f"{'resolved as a brawler':<30} {len(resolved)}")
print(f"{'mentions added':<30} {added}")