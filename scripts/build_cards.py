"""Parse every downloaded brawler page into a card: data/fandom/brawlers/<id>.wiki -> MongoDB brawler_cards.
Run: python scripts/build_cards.py"""
import datetime as dt
from collections import Counter
from pathlib import Path

import yaml
from pymongo import MongoClient

from draft_coach.cards import parse_card
from draft_coach.config import MONGO_URI
from draft_coach.names import load_brawlers
from draft_coach.sources import load_sources

PAGES = Path("data/fandom/brawlers")
# History window: buffs and nerfs from the oldest corpus source on (the period the pro advice covers)
since = min(dt.date.fromisoformat(str(s["published"])) for s in load_sources(Path("data/sources.yaml")) if "published" in s)
collection = MongoClient(MONGO_URI)["draft_coach"]["brawler_cards"]

cards = []
for brawler in load_brawlers(Path("data/brawlers.yaml")):
    path = PAGES / f"{brawler['id']}.wiki"
    card = {"_id": brawler["id"], "name": brawler["en"],
            "url": f"https://brawlstars.fandom.com/wiki/{brawler['en'].replace(' ', '_')}",
            "license": "CC BY-SA", "fetched_at": dt.date.fromtimestamp(path.stat().st_mtime).isoformat(),
            **parse_card(path.read_text(encoding="utf-8"), since)}
    collection.replace_one({"_id": card["_id"]}, card, upsert=True)  # idempotent: rerun = same collection
    cards.append(card)


def requirements(card):
    """Every condition written as "(with ...)" on the card."""
    values = [v for a in card["abilities"] for vals in a.get("stats", {}).values() if isinstance(vals, list) for v in vals]
    return [r for v in values + card["movement_speed"] for r in v.get("requires", [])]


# End-of-step check
types = Counter(a["type"] for c in cards for a in c["abilities"])
per_card = {c["_id"]: Counter(a["type"] for a in c["abilities"]) for c in cards}
reqs = [r for c in cards for r in requirements(c)]
unread = [(c["_id"], v["raw"]) for c in cards for a in c["abilities"] for vals in a.get("stats", {}).values()
          if isinstance(vals, list) for v in vals if "raw" in v]
print(f"{'cards (table / MongoDB)':<34} {len(cards)} / {collection.count_documents({})}")
print(f"{'abilities by type':<34} {dict(types)}")
print(f"{'not 2 gadgets / 2 star powers':<34} {[k for k, c in per_card.items() if c['gadget'] != 2 or c['star_power'] != 2] or 'none'}")
print(f"{'without hypercharge':<34} {[k for k, c in per_card.items() if not c['hypercharge']]}")
print(f"{'buffies':<34} {sum(1 for c in cards for a in c['abilities'] if a['buffie'])}")
print(f"{'conditions by type':<34} {dict(Counter(next(iter(r)) for r in reqs))}")
print(f"{'states (not an ability)':<34} {Counter(r['state'] for r in reqs if 'state' in r).most_common(8)}")
print(f"{'values kept as text':<34} {len(unread)} e.g. {unread[:3]}")
print(f"{'history since ' + since.isoformat():<34} {sum(len(c['history']) for c in cards)} buff/nerf lines, max {max(len(c['history']) for c in cards)} per brawler")
brock = next(c for c in cards if c["_id"] == "brock")
short = {**brock, "abilities": [{**a, "text": a["text"][:80] + "..."} for a in brock["abilities"][:3]], "history": brock["history"][:2]}
print("\nSample card (abridged):\n" + yaml.safe_dump(short, sort_keys=False, allow_unicode=True, width=120))
assert len(cards) == collection.count_documents({}), "MongoDB has cards that are not in the table"