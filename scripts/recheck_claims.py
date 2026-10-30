"""Step 6: re-apply the guards of claims.py to the saved raw answers, without asking the model again;
then list the likely misspellings behind "brawler_not_in_section" (alias candidates for brawlers.yaml).
Run: python scripts/recheck_claims.py"""
from pathlib import Path

from pymongo import MongoClient

from draft_coach.claim_store import alias_candidates, print_report, save_section
from draft_coach.claims import build_prompt, corpus_sections, section_hash
from draft_coach.config import MONGO_URI
from draft_coach.maps import load_maps
from draft_coach.names import NameNormalizer, load_brawlers
from draft_coach.sources import load_sources

MODEL = "gemini-3.8-flash"  # same as extract_claims.py: part of the section key

db = MongoClient(MONGO_URI)["draft_coach"]
sources = {s["id"]: s for s in load_sources(Path("data/sources.yaml"))}
all_maps = load_maps(Path("data/maps.yaml"))
pool_maps = [m for m in all_maps if m["in_pool"]]
brawler_list = load_brawlers(Path("data/brawlers.yaml"))
normalizer = NameNormalizer(brawler_list)
wiki_class = {c["_id"]: c.get("wiki_class") for c in db["brawler_cards"].find({}, {"wiki_class": 1})}
brawler_info = {b["id"]: {"class": b["class"], "wiki_class": wiki_class.get(b["id"])} for b in brawler_list}

sections = corpus_sections(list(db["normalized"].find()), sources)
for section in sections:
    section["key"] = section_hash(build_prompt(section["source"], section, pool_maps), MODEL)
by_key = {s["key"]: s for s in sections}

rechecked = without_raw = 0
for doc in db["claim_sections"].find({"_id": {"$in": list(by_key)}}):
    if "raw_claims" not in doc:  # saved before raw answers were kept: extract_claims.py asks it again
        without_raw += 1
        continue
    save_section(db, by_key[doc["_id"]], doc["raw_claims"], doc["model"], all_maps, normalizer, brawler_info)
    rechecked += 1
print(f"{'sections rechecked':<28} {rechecked} ({without_raw} without raw answer: re-asked by extract_claims.py)")
print_report(db, normalizer, len(sections))

brawlers = {b["id"]: b for b in brawler_list}
print(f"\n{'alias candidates':<28} brawler <- word of the quote (count); '-' = brawler not in the quote")
for (brawler, word), n in alias_candidates(db, by_key, normalizer, brawlers, all_maps).most_common(40):
    print(f"  {brawlers[brawler]['en']:<20} <- {word:<20} {n}")