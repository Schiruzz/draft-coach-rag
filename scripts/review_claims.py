"""Step 6: blind review of 40 claims, stratified by source, to measure whether the claims are faithful to the source.
First run: writes data/eval/claims_review.yaml (2 claims per source, the rest at random; fixed seed).
Federico fills faithful / standalone / labels_ok with yes or no (and a note when useful).
Next runs: print the share of yes per question, with its 95% interval (the file is never overwritten).
Run: python scripts/review_claims.py"""
import random
from collections import defaultdict
from pathlib import Path

import yaml
from pymongo import MongoClient

from draft_coach.config import MONGO_URI
from draft_coach.recall import wilson_interval

REVIEW_FILE = Path("data/eval/claims_review.yaml")
SAMPLE_SIZE = 40
PER_SOURCE = 2
QUESTIONS = {"faithful": "the source says this rule (not more, not the opposite)",
             "standalone": "the concept makes sense without the video",
             "labels_ok": "sides, direction and the other labels are right"}
SHOWN = ("concept", "reason", "example", "topic", "relation", "brawlers", "classes", "effects", "against_brawlers",
         "against_classes", "against_effects", "draft_phase", "rank_band", "modes", "map_ids", "lane_role", "mode_role")

if not REVIEW_FILE.exists():
    claims = list(MongoClient(MONGO_URI)["draft_coach"]["claims"].find())
    rng = random.Random(0)  # fixed seed: the same sample until the claims change
    by_source = defaultdict(list)
    for c in claims:
        by_source[c["source_id"]].append(c)
    sample = [c for source in sorted(by_source) for c in rng.sample(by_source[source], min(PER_SOURCE, len(by_source[source])))]
    rest = [c for c in claims if c not in sample]
    sample += rng.sample(rest, SAMPLE_SIZE - len(sample))
    entries = [{"id": c["_id"], "source": c["source_id"], "link": c["url"], "quote": c["quote"],
                **{k: c[k] for k in SHOWN if c[k]}, **{q: None for q in QUESTIONS}, "note": ""}
               for c in sorted(sample, key=lambda c: (c["source_id"], c["start_s"]))]
    header = "# Claim review: answer yes or no.\n" + "".join(f"# {q}: {text}\n" for q, text in QUESTIONS.items())
    REVIEW_FILE.write_text(header + yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, width=120),
                           encoding="utf-8")
    print(f"{REVIEW_FILE} written: {len(entries)} claims from {len(by_source)} sources")
else:
    entries = yaml.safe_load(REVIEW_FILE.read_text(encoding="utf-8"))
    todo = sum(any(e[q] is None for q in QUESTIONS) for e in entries)
    if todo:
        print(f"{todo} of {len(entries)} claims still to review in {REVIEW_FILE}")
    else:
        print(f"{'question':<12} {'yes':>8} {'95% interval':>14}")
        for q in QUESTIONS:
            hits = sum(e[q] == "yes" or e[q] is True for e in entries)
            low, high = wilson_interval(hits, len(entries))
            print(f"{q:<12} {hits / len(entries):>8.0%} {f'{low:.0%}-{high:.0%}':>14}")