"""Step 6: re-apply the guards of claims.py to the saved raw answers, without asking the model again;
then list the likely misspellings behind "brawler_not_in_section" (alias candidates for brawlers.yaml).
Run: python scripts/recheck_claims.py"""
from pathlib import Path

from pymongo import MongoClient

from draft_coach.claim_store import alias_candidates, print_report, save_section
from draft_coach.claims import PASSIVE_MAPS, build_prompt, corpus_sections, section_hash, unsaid_brawlers
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
brawler_names = [b["en"] for b in brawler_list]  # same prompt as extract_claims.py
wiki_class = {c["_id"]: c.get("wiki_class") for c in db["brawler_cards"].find({}, {"wiki_class": 1})}
brawler_info = {b["id"]: {"class": b["class"], "wiki_class": wiki_class.get(b["id"])} for b in brawler_list}

sections = corpus_sections(list(db["normalized"].find()), sources)
for section in sections:
    section["key"] = section_hash(build_prompt(section["source"], section, pool_maps, brawler_names), MODEL)
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

# Claims flagged by the swap guard, to compare with the manual review (step 7c)
print(f"\n{'name swaps flagged':<28} source, minute, brawler of the rule -> brawler of the quote, concept")
for c in db["claims"].find({"name_swaps.0": {"$exists": True}}).sort([("source_id", 1), ("start_s", 1)]):
    swaps = ", ".join(f"{rule} -> {quote}" for rule, quote in c["name_swaps"])
    print(f"  {c['source_id']:<34} {c['start_s'] // 60:>3}:{c['start_s'] % 60:02d}  {swaps:<22} {c['concept']}")


# Claims whose brawler is never said near the quote (step 7c), for three windows: the window is fixed on this measure.
# The flagged sets are nested (a longer window reads more text), so one list shows every window.
WINDOWS_S = [20, 40, 60]
segments_of = {d["_id"]: d["segments"] for d in db["normalized"].find({}, {"segments": 1})}
map_names = [m["name"] for m in all_maps] + [a for m in all_maps for a in m["aliases"]] + PASSIVE_MAPS


def chapter_at(source: dict, start_s: int) -> str | None:
    titles = [c["title"] for c in source.get("chapters", []) if c["start_s"] <= start_s]
    return titles[-1] if titles else None


unsaid = []  # (claim, brawlers never said within the shortest window, longest window that still flags it)
for c in db["claims"].find().sort([("source_id", 1), ("start_s", 1)]):
    found = {w: unsaid_brawlers(c, segments_of[c["source_id"]], chapter_at(sources[c["source_id"]], c["start_s"]),
                                normalizer, map_names, w) for w in WINDOWS_S}
    if found[WINDOWS_S[0]]:
        unsaid.append((c, found[WINDOWS_S[0]], max(w for w in WINDOWS_S if found[w])))
print(f"\n{'never said near the quote':<28} window: counters videos / other formats")
for w in WINDOWS_S:
    counters = sum(1 for c, _, longest in unsaid if longest >= w and sources[c["source_id"]]["format"] == "counters")
    print(f"  {w:>3} s {counters:>22} / {sum(1 for _, _, longest in unsaid if longest >= w) - counters}")
print(f"\n{'never said near the quote':<28} source, minute, up to window, brawlers, concept")
for c, brawler_ids, longest in unsaid:
    print(f"  {c['source_id']:<34} {c['start_s'] // 60:>3}:{c['start_s'] % 60:02d}  {longest:>3} s  "
          f"{', '.join(brawler_ids):<20} {c['concept']}")

brawlers = {b["id"]: b for b in brawler_list}
print(f"\n{'alias candidates':<28} brawler <- word of the quote (count); '-' = brawler not in the quote")
for (brawler, word), n in alias_candidates(db, by_key, normalizer, brawlers, all_maps).most_common(40):
    print(f"  {brawlers[brawler]['en']:<20} <- {word:<20} {n}")