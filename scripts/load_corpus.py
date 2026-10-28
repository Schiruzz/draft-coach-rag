"""Load every source in data/sources.yaml into MongoDB (db draft_coach, collection sources).
Run: python scripts/load_corpus.py"""
from pathlib import Path

import yaml
from pymongo import MongoClient

from draft_coach.config import MONGO_URI
from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers
from draft_coach.screens import screen_segments
from draft_coach.sources import load_sources
from draft_coach.transcript import parse_transcript
from draft_coach.wiki_chunks import page_sections

DATA = Path("data")
RAW_FILES = {"transcript": ("transcripts", ".txt"), "wiki": ("fandom", ".wiki")}  # type -> (folder, extension)

sources = load_sources(DATA / "sources.yaml")
collection = MongoClient(MONGO_URI)["draft_coach"]["sources"]

for source in sources:
    folder, extension = RAW_FILES[source["type"]]
    raw = (DATA / folder / f"{source['id']}{extension}").read_text(encoding="utf-8")
    document = {key: value for key, value in source.items() if key != "id"}
    if source["type"] == "transcript":
        # The transcript panel repeats chapter titles as text lines: drop them,
        # chapters are already in sources.yaml with their timestamps
        chapter_titles = {chapter["title"] for chapter in source.get("chapters", [])}
        raw = "\n".join(line for line in raw.splitlines() if line.strip() not in chapter_titles)
        document["segments"] = parse_transcript(raw)
        # Lists shown on screen but not read aloud (data/screens/<id>.yaml): added as segments at their chapter
        screens = DATA / "screens" / f"{source['id']}.yaml"
        if screens.exists():
            extra = screen_segments(yaml.safe_load(screens.read_text(encoding="utf-8")), source["chapters"],
                                    load_maps(DATA / "maps.yaml"), load_brawlers(DATA / "brawlers.yaml"))
            # appended, not sorted in: segment indices are part of the name-candidate ids, so they must not shift
            document["segments"] = document["segments"] + extra
        size = f"{len(document['segments'])} segments"
    else:
        # Raw wikitext by section: the wiki layer (step 5b) picks its sections, nothing is dropped here
        document["sections"] = page_sections(raw)
        size = f"{len(document['sections'])} sections"
    if not document.get("segments") and not document.get("sections"):
        raise ValueError(f"{source['id']}: raw file has no content ({folder}/{source['id']}{extension})")
    collection.replace_one({"_id": source["id"]}, document, upsert=True)  # idempotent: rerun = same database
    print(f"{source['id']:<30} {source['type']:<11} {size}")

# End-of-step check: listed sources = raw files = documents in MongoDB
n_files = sum(len(list((DATA / folder).glob(f"*{extension}"))) for folder, extension in RAW_FILES.values())
n_documents = collection.count_documents({})
print(f"\n{'sources listed':<30} {len(sources)}")
print(f"{'raw files':<30} {n_files}")
print(f"{'documents in MongoDB':<30} {n_documents}")
assert len(sources) == n_files == n_documents, "sources.yaml, raw files and MongoDB disagree"