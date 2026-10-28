"""Chunks of the wiki layer (step 5b): mode pages (sources) and map Layouts (maps) -> chunks, in index A.
Run: python scripts/chunk_wiki.py   then: python scripts/embed_chunks.py"""
from pathlib import Path
from statistics import median

from pymongo import MongoClient

from draft_coach.chunks import METHOD
from draft_coach.config import MONGO_URI
from draft_coach.names import NameNormalizer, load_brawlers
from draft_coach.wiki_chunks import INTRO, LAYER, layout_text, mode_texts

normalizer = NameNormalizer(load_brawlers(Path("data/brawlers.yaml")))
db = MongoClient(MONGO_URI)["draft_coach"]

chunks = []
for source in db["sources"].find({"type": "wiki"}):
    mode = source["_id"].removeprefix("fandom-").replace("-", "_")  # fandom-hot-zone -> hot_zone, as in maps.yaml
    for section, text in mode_texts(source["title"], source["sections"]):
        chunks.append({"source_id": source["_id"], "section": section, "mode": mode, "text": text,
                       "url": source["url"] + ("" if section == INTRO else "#Tips")})
for m in db["maps"].find({"in_pool": True}):  # the quiz uses only the maps in the pool
    chunks.append({"source_id": f"fandom-map-{m['_id']}", "section": "Layout", "mode": m["mode"], "map_id": m["_id"],
                   "text": layout_text(m), "url": m["url"] + "#Layout"})

# Mentions with official names only: the wiki text is already correct, no aliases to fix (decided 28/10)
docs = [{"_id": f"{c['source_id']}:{METHOD}:{c['section']}", "method": METHOD, "layer": LAYER,
         **c, "mentions": normalizer.mentions(c["text"])} for c in chunks]
# Rebuilt on every run (about 40 chunks, instant); ids are stable, so embed_chunks.py embeds only changed texts
db["chunks"].delete_many({"layer": LAYER})
db["chunks"].insert_many(docs)

# End-of-step check: every mode page has rules and tips, every pool map has its Layout
print(f"{'source':<32} {'chunks':>6} {'words':>10} {'mentions':>9}")
for source_id in sorted({d["source_id"] for d in docs}):
    mine = [d for d in docs if d["source_id"] == source_id]
    words = [len(d["text"].split()) for d in mine]
    print(f"{source_id:<32} {len(mine):>6} {f'{min(words)}-{max(words)}':>10} {sum(len(d['mentions']) for d in mine):>9}")
words = sorted(len(d["text"].split()) for d in docs)
print(f"\n{'wiki chunks':<32} {db['chunks'].count_documents({'layer': LAYER})}   "
      f"words min / median / max {words[0]} / {median(words):.0f} / {words[-1]}")
sample = next(d for d in docs if d["section"] == "Tips 1")
print(f"\nSample {sample['_id']}  mentions {sample['mentions']}\n{sample['text'][:400]}...")
wiki_sources = {s["_id"] for s in db["sources"].find({"type": "wiki"})}
assert {d["source_id"] for d in docs if d["section"] == INTRO} == wiki_sources, "a mode page has no rules"
assert {d["source_id"] for d in docs if d["section"].startswith("Tips")} == wiki_sources, "a mode page has no tips"
assert sum(d["section"] == "Layout" for d in docs) == db["maps"].count_documents({"in_pool": True}), "a map has no Layout"