"""Step 6: extract the claims of every section of the corpus and save them in MongoDB (sources -> claims).
Incremental and resumable: a section is asked once per (version, model, prompt); each raw answer is saved as soon
as it arrives, so Ctrl+C or a failed section loses nothing (failed sections are asked again on the next run).
Guards changed later are re-applied without the model: scripts/recheck_claims.py.
Sections that no longer exist (changed transcript, new version) are removed with their claims.
Run: python scripts/extract_claims.py"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from pymongo import MongoClient
from tqdm import tqdm

from draft_coach.claim_store import print_report, save_section
from draft_coach.claims import Claim, build_prompt, corpus_sections, section_hash
from draft_coach.config import MONGO_URI
from draft_coach.gemini import generate_json_hedged
from draft_coach.maps import load_maps
from draft_coach.names import NameNormalizer, load_brawlers
from draft_coach.sources import load_sources

MODEL, THINKING = "gemini-3.8-flash", "low"  # chosen on the extract-v1 sample
PARALLEL_CALLS = 8
TIMEOUT_MS = 120_000  # server queue on Flash in "eu": one section measured at 97 s with 9 claims (29/10); p90 78 s (28/10)
HEDGE_AFTER_S = 30  # backup request after twice the slowest normal section of the v4 sample (8-15 s)

db = MongoClient(MONGO_URI)["draft_coach"]
sources = {s["id"]: s for s in load_sources(Path("data/sources.yaml"))}
all_maps = load_maps(Path("data/maps.yaml"))
pool_maps = [m for m in all_maps if m["in_pool"]]  # listed in the prompt
brawler_list = load_brawlers(Path("data/brawlers.yaml"))
normalizer = NameNormalizer(brawler_list)
brawler_names = [b["en"] for b in brawler_list]  # listed in the prompt
wiki_class = {c["_id"]: c.get("wiki_class") for c in db["brawler_cards"].find({}, {"wiki_class": 1})}
brawler_info = {b["id"]: {"class": b["class"], "wiki_class": wiki_class.get(b["id"])} for b in brawler_list}

# Every section with its prompt and key; the key changes with the text, the source data, the prompt or the model
sections = corpus_sections(list(db["normalized"].find()), sources)
for section in sections:
    section["prompt"] = build_prompt(section["source"], section, pool_maps, brawler_names)
    section["key"] = section_hash(section["prompt"], MODEL)
keys = [s["key"] for s in sections]

# Stale sections (old version, changed transcript) go away with their claims: the collections mirror the corpus
stale = db["claim_sections"].delete_many({"_id": {"$nin": keys}}).deleted_count
db["claims"].delete_many({"section_key": {"$nin": keys}})
# done = saved with the raw answer; sections saved before raw answers were kept are asked again
done = {d["_id"] for d in db["claim_sections"].find({"raw_claims": {"$exists": True}}, {"_id": 1})}
todo = [s for s in sections if s["key"] not in done]
print(f"{'sections':<20} {len(sections)} ({len(done)} done, {len(todo)} to ask, {stale} stale removed)")


backups = {"sent": 0, "won": 0}  # how often the backup request was needed and how often it answered first


def ask(section: dict) -> list[Claim] | None:
    claims, sent, won = generate_json_hedged(client, MODEL, section["prompt"], list[Claim], THINKING, HEDGE_AFTER_S,
                                             label=f"{section['source']['id']} {section['start_s']}")
    backups["sent"] += sent
    backups["won"] += won
    return claims


client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))  # project and location from .env
failed = 0
with ThreadPoolExecutor(PARALLEL_CALLS) as pool, tqdm(total=len(todo), desc="sections") as bar:
    futures = {pool.submit(ask, s): s for s in todo}
    try:
        for future in as_completed(futures):
            claims = future.result()
            bar.update()
            if claims is None:  # still failing after all attempts: asked again on the next run
                failed += 1
                continue
            save_section(db, futures[future], [c.model_dump() for c in claims], MODEL, all_maps, normalizer, brawler_info)
    except KeyboardInterrupt:  # Ctrl+C: drop the queued calls; saved sections stay saved
        pool.shutdown(cancel_futures=True)
        raise

print(f"\n{'failed in this run':<28} {failed}")
print(f"{'backup requests':<28} {backups['sent']} sent, {backups['won']} answered first (this run)")
print_report(db, normalizer, len(sections))