"""Step 6, before the full run: extract claims from 5 sections, to check the prompt and size the run.
Prints the number of sections of the corpus, then time and claims per section; writes
reports/extraction_sample.md for Federico to judge the claims. Saves nothing in MongoDB.
Run: python scripts/sample_extraction.py"""
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai
from google.genai import types
from pymongo import MongoClient

from draft_coach.claims import EXTRACT_VERSION, SECTION_WORDS, Claim, build_prompt, check_claim, corpus_sections
from draft_coach.config import MONGO_URI
from draft_coach.gemini import generate_json
from draft_coach.maps import load_maps
from draft_coach.names import NameNormalizer, load_brawlers
from draft_coach.sources import load_sources

MODEL, THINKING = "gemini-3.8-flash", "low"  # chosen on the extract-v1 sample: Flash-Lite turned brawlers into classes
SAMPLE_FORMATS = ["guide", "draft_review", "counters", "tier_list", "map_cheat_sheet"]  # one section per format
TIMEOUT_MS = 180_000  # generous on purpose: this run measures the time a section takes

db = MongoClient(MONGO_URI)["draft_coach"]
sources = {s["id"]: s for s in load_sources(Path("data/sources.yaml"))}
pool_maps = [m for m in load_maps(Path("data/maps.yaml")) if m["in_pool"]]
brawler_list = load_brawlers(Path("data/brawlers.yaml"))
normalizer = NameNormalizer(brawler_list)
brawler_names = [b["en"] for b in brawler_list]  # listed in the prompt
wiki_class = {c["_id"]: c.get("wiki_class") for c in db["brawler_cards"].find({}, {"wiki_class": 1})}
brawler_info = {b["id"]: {"class": b["class"], "wiki_class": wiki_class.get(b["id"])} for b in brawler_list}

# All sections of the corpus: their number is the number of calls of the full run
sections = corpus_sections(list(db["normalized"].find()), sources)
words = [len(s["text"].split()) for s in sections]
print(f"{'sections in the corpus':<25} {len(sections)} (max {SECTION_WORDS} words; "
      f"min / median / max {min(words)} / {sorted(words)[len(words) // 2]} / {max(words)})")

rng = random.Random(0)  # fixed seed: the same sample as extract-v1, so the two versions are compared on the same text
sample = [rng.choice([s for s in sections if s["source"]["format"] == f]) for f in SAMPLE_FORMATS]


def extract(section: dict) -> tuple[float, list[Claim] | None]:
    start = time.monotonic()
    claims = generate_json(client, MODEL, build_prompt(section["source"], section, pool_maps, brawler_names), list[Claim], THINKING,
                           label=f"{section['source']['id']} {section['start_s']}")
    return time.monotonic() - start, claims


client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))  # project and location from .env
with ThreadPoolExecutor(len(sample)) as pool:  # the 5 sections at once: wall time = slowest call
    results = list(pool.map(extract, sample))

# Table: time and claims kept / rejected per section; report with every field of every claim
print(f"\n{EXTRACT_VERSION}, {MODEL}\n{'section':<45} {'seconds':>8} {'kept':>5} {'rejected':>9}")
report, reasons, mode_roles = [f"# Extraction sample ({EXTRACT_VERSION}, {MODEL})", ""], Counter(), Counter()
for section, (seconds, claims) in zip(sample, results):
    source = section["source"]
    name = f"{source['id']} {section['start_s']}-{section['end_s']} s"
    report += [f"## {name} ({source['format']}, {len(section['text'].split())} words, {seconds:.0f} s)",
               f"Chapter: {section['chapter']}. [Video]({source['url']}&t={section['start_s']}s)", ""]
    if claims is None:
        print(f"{name:<45} {seconds:>8.0f} {'failed':>15}")
        report += ["Failed after all attempts.", ""]
        continue
    checked = [check_claim(c, section, pool_maps, normalizer, brawler_info) for c in claims]
    kept = [record for record, _ in checked if record]
    print(f"{name:<45} {seconds:>8.0f} {len(kept):>5} {len(claims) - len(kept):>9}")
    for claim, (record, reason) in zip(claims, checked):
        if record is None:
            reasons[reason] += 1
            report.append(f"- REJECTED ({reason}): {claim.concept} — \"{claim.quote}\"")
            continue
        mode_roles.update(record["mode_role"])
        filled = {k: v for k, v in record.items()  # only the fields the model or the code filled
                  if v and not (isinstance(v, dict) and not any(v.values()))
                  and k not in {"concept", "quote", "start_s", "end_s", "relation"}}
        if record["relation"] != "none":
            filled = {"relation": record["relation"], **filled}  # the direction first: it reads the two sides
        report += [f"- **{record['concept']}** [{record['start_s']} s]({source['url']}&t={record['start_s']}s)",
                   f"  \"{record['quote']}\"",
                   f"  {'; '.join(f'{k}: {v}' for k, v in filled.items())}"]
    report.append("")

print(f"\n{'rejected by reason':<25} {dict(reasons) or '-'}")
print(f"{'mode_role labels':<25} {dict(mode_roles) or '-'}")
Path("reports").mkdir(exist_ok=True)
Path("reports/extraction_sample.md").write_text("\n".join(report), encoding="utf-8")
print("report: reports/extraction_sample.md")