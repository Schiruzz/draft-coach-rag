"""Load every source in data/sources.yaml into MongoDB (db draft_coach, collection sources).
Run: python scripts/load_corpus.py"""
from pathlib import Path

import mwparserfromhell
from pymongo import MongoClient

from draft_coach.config import MONGO_URI
from draft_coach.sources import load_sources
from draft_coach.transcript import parse_transcript

DATA = Path("data")
RAW_FILES = {"transcript": ("transcripts", ".txt"), "wiki": ("fandom", ".wiki")}  # type -> (folder, extension)

sources = load_sources(DATA / "sources.yaml")
collection = MongoClient(MONGO_URI)["draft_coach"]["sources"]

for source in sources:
    folder, extension = RAW_FILES[source["type"]]
    raw = (DATA / folder / f"{source['id']}{extension}").read_text(encoding="utf-8")
    document = {key: value for key, value in source.items() if key != "id"}
    if source["type"] == "transcript":
        document["segments"] = parse_transcript(raw)
        size = f"{len(document['segments'])} segments"
    else:
        document["text"] = mwparserfromhell.parse(raw).strip_code().strip()
        size = f"{len(document['text'])} chars"
    if not document.get("segments") and not document.get("text"):
        raise ValueError(f"{source['id']}: raw file has no content ({folder}/{source['id']}{extension})")
    collection.replace_one({"_id": source["id"]}, document, upsert=True)  # idempotent: rerun = same database
    print(f"{source['id']:<22} {source['type']:<11} {size}")

# End-of-step check: listed sources = raw files = documents in MongoDB
n_files = sum(len(list((DATA / folder).glob(f"*{extension}"))) for folder, extension in RAW_FILES.values())
n_documents = collection.count_documents({})
print(f"\n{'sources listed':<22} {len(sources)}")
print(f"{'raw files':<22} {n_files}")
print(f"{'documents in MongoDB':<22} {n_documents}")
assert len(sources) == n_files == n_documents, "sources.yaml, raw files and MongoDB disagree"