"""Parse every downloaded map page: data/fandom/maps/<id>.wiki -> MongoDB maps.
Run: python scripts/build_maps.py"""
import datetime as dt
from collections import Counter
from pathlib import Path

import yaml
from pymongo import MongoClient

from draft_coach.config import MONGO_URI
from draft_coach.map_pages import parse_map_page
from draft_coach.maps import FEATURE_TILES, load_maps

PAGES = Path("data/fandom/maps")
collection = MongoClient(MONGO_URI)["draft_coach"]["maps"]

docs = []
for m in load_maps(Path("data/maps.yaml")):
    path = PAGES / f"{m['id']}.wiki"
    doc = {"_id": m["id"], "name": m["name"], "mode": m["mode"], "in_pool": m["in_pool"],
           "types": m["types"], "features": m["features"], "notes": m.get("notes", ""),           
           "url": f"https://brawlstars.fandom.com/wiki/{m['name'].replace(' ', '_')}", "license": "CC BY-SA",
           "fetched_at": dt.date.fromtimestamp(path.stat().st_mtime).isoformat(),
           "image_url": (PAGES / f"{m['id']}.image_url").read_text(encoding="utf-8").strip(),  # shown with credit
           **parse_map_page(path.read_text(encoding="utf-8"))}
    collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)  # idempotent: rerun = same collection
    docs.append(doc)

# End-of-step check
wrong_mode = [(d["_id"], d["event"]) for d in docs if (d["event"] or "").lower().replace(" ", "_") != d["mode"]]
tile_keys = Counter(k for d in docs for k in d["tiles"])
# A feature that Fandom can confirm needs at least one tile of its kind (e.g. water_split -> Lake)
unconfirmed = [(d["_id"], f) for d in docs for f in d["features"] if f in FEATURE_TILES and not d["tiles"].get(FEATURE_TILES[f])]
print(f"{'maps (table / MongoDB)':<26} {len(docs)} / {collection.count_documents({})}")
print(f"{'event != mode':<26} {wrong_mode or 'none'}")
print(f"{'empty layout':<26} {[d['_id'] for d in docs if not d['layout']] or 'none'}")
print(f"{'tile keys (maps with it)':<26} {dict(tile_keys.most_common())}")
print(f"{'other infobox fields':<26} {[(d['_id'], d['other']) for d in docs if d['other']] or 'none'}")
print(f"{'features without tiles':<26} {unconfirmed or 'none'}")
print(f"{'history lines':<26} {sum(len(d['history']) for d in docs)}, max {max(len(d['history']) for d in docs)} per map")
sample = next(d for d in docs if d["_id"] == "hot-potato")
print("\nSample (abridged):\n" + yaml.safe_dump({**sample, "history": sample["history"][-3:]}, sort_keys=False, allow_unicode=True, width=120))
assert len(docs) == collection.count_documents({}), "MongoDB has maps that are not in the table"
assert not unconfirmed, "a feature in maps.yaml is not supported by the Fandom tiles"