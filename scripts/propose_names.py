"""Ask Gemini which brawler (if any) each ambiguous name in name_candidates means, then write the review files.
Run: python scripts/propose_names.py"""
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from google import genai
from google.genai import types
from pydantic import BaseModel
from pymongo import MongoClient
from tqdm import tqdm

from draft_coach.config import GEMINI_MODEL, MONGO_URI  # importing config also loads GEMINI_API_KEY from .env
from draft_coach.names import load_brawlers

PROMPT_VERSION = "v2"  # bump when the prompt changes: proposals of older versions are asked again
BATCH_SIZE = 25
PARALLEL_CALLS = 8
REVIEW_FILE = Path("data/name_review/decisions.yaml")  # low-confidence answers: Federico decides each one
SAMPLE_FILE = Path("data/name_review/sample.yaml")     # blind sample of high-confidence answers: measures Gemini
SAMPLE_SIZE = 60
PROMPT = """You check automatic YouTube transcripts of Brawl Stars videos.
In each item, the speech recognizer may have written the word in "alias" instead of a brawler name.
"options" lists the brawlers it could be (id: name).
Read the context (previous segment, [current segment], next segment) and answer with the id of the brawler
that "alias" means, or null if it is not one of them. Answer for every item id.
Use confidence "low" when the context is not enough to be sure.

Items:
{items}"""


class Proposal(BaseModel):
    id: str
    brawler: str | None  # one of the item's options, or None
    confidence: Literal["high", "low"]


db = MongoClient(MONGO_URI)["draft_coach"]
en_name = {b["id"]: b["en"] for b in load_brawlers(Path("data/brawlers.yaml"))}
segments = {doc["_id"]: doc["segments"] for doc in db["normalized"].find()}
urls = {s["_id"]: s["url"] for s in db["sources"].find({}, {"url": 1})}


def context(candidate: dict) -> str:
    """Previous, current (in brackets) and next segment of the candidate."""
    texts = [s["text"] for s in segments[candidate["source_id"]]]
    i = candidate["segment"]
    return " ".join([*texts[max(i - 1, 0):i], f"[{texts[i]}]", *texts[i + 1:i + 2]])


# Resumable: candidates already answered with this prompt version are skipped
done = {p["_id"] for p in db["name_proposals"].find({"prompt_version": PROMPT_VERSION}, {"_id": 1})}
todo = [c for c in db["name_candidates"].find() if c["_id"] not in done]
batches = [todo[start:start + BATCH_SIZE] for start in range(0, len(todo), BATCH_SIZE)]
client = genai.Client(http_options=types.HttpOptions(
    timeout=30_000,  # ms: a stuck call fails instead of hanging
    retry_options=types.HttpRetryOptions(attempts=3),  # retries 429/5xx with growing waits
))
print(f"{'candidates to ask':<30} {len(todo)} ({len(done)} already answered)")


def ask(batch: list[dict]) -> tuple[list[dict], list[Proposal]]:
    """One Gemini call for one batch; runs in a worker thread."""
    items = [{"id": c["_id"], "alias": c["alias"], "options": {b: en_name[b] for b in c["brawler_ids"]},
              "context": context(c)} for c in batch]
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=PROMPT.format(items=yaml.safe_dump(items, allow_unicode=True, sort_keys=False)),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=list[Proposal], temperature=0,
            thinking_config=types.ThinkingConfig(thinking_level="low")),  # a short choice with context needs little reasoning
    )
    return batch, response.parsed


with ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as pool:
    # MongoDB writes stay in the main thread; results arrive in batch order
    for batch, parsed in tqdm(pool.map(ask, batches), total=len(batches), desc="Gemini batches", unit="batch"):
        by_id = {c["_id"]: c for c in batch}
        for proposal in parsed:
            candidate = by_id.get(proposal.id)
            # Guards: Gemini may alter an id or answer outside the options; skipped answers are asked again
            if candidate is None or proposal.brawler not in [*candidate["brawler_ids"], None]:
                continue
            db["name_proposals"].replace_one({"_id": proposal.id}, {
                "source_id": candidate["source_id"], "segment": candidate["segment"],
                "start_s": candidate["start_s"], "alias": candidate["alias"], "options": candidate["brawler_ids"],
                "brawler_id": proposal.brawler, "confidence": proposal.confidence,
                "model": GEMINI_MODEL, "prompt_version": PROMPT_VERSION, "created": datetime.now(timezone.utc)},
                upsert=True)


def review_entry(proposal: dict) -> dict:
    """What Federico needs to judge a candidate: the name, the context and the video at that second."""
    return {"id": proposal["_id"], "alias": proposal["alias"], "options": proposal["options"],
            "link": f"{urls[proposal['source_id']]}&t={proposal['start_s']}s", "context": context(proposal)}


proposals = sorted(db["name_proposals"].find({"prompt_version": PROMPT_VERSION}), key=lambda p: p["_id"])
high = [p for p in proposals if p["confidence"] == "high"]
n_candidates = db["name_candidates"].count_documents({})
if len(proposals) < n_candidates:
    print(f"\n{n_candidates - len(proposals)} candidates without answer: run the script again")
else:
    # Decisions already taken are kept (Federico's work is never lost) and their entries stay in the file
    previous = {e["id"]: e["decision"] for e in yaml.safe_load(REVIEW_FILE.read_text(encoding="utf-8"))} \
        if REVIEW_FILE.exists() else {}
    to_review = [p for p in proposals if p["confidence"] == "low" or p["_id"] in previous]
    # decision: a brawler id, or "none" if the alias is not a brawler; null = still to decide
    review = [{**review_entry(p), "proposed": p["brawler_id"] or "none", "decision": previous.get(p["_id"])}
              for p in to_review]
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text(yaml.safe_dump(review, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    # Blind: Gemini's answer is not shown, so it cannot anchor the label; fixed seed = same sample on every run
    if SAMPLE_FILE.exists():  # never overwrite Federico's labels
        print(f"\n{SAMPLE_FILE} already exists: not overwritten (delete it to regenerate)")
    else:
        sample = [{**review_entry(p), "label": None} for p in random.Random(0).sample(high, SAMPLE_SIZE)]
        SAMPLE_FILE.write_text(yaml.safe_dump(sample, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")

# End-of-step check
print(f"\n{'candidates':<30} {n_candidates}")
print(f"{'proposals (' + PROMPT_VERSION + ')':<30} {len(proposals)}")
print(f"{'  is a brawler':<30} {sum(p['brawler_id'] is not None for p in proposals)}")
print(f"{'  low confidence':<30} {len(proposals) - len(high)}")
print(f"{'  high confidence':<30} {len(high)} (blind sample: {SAMPLE_SIZE})")