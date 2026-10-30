"""Claims in MongoDB: one record per section (with the raw answer of the model) and one document per kept claim.
The raw answer is kept so the guards can be re-applied without asking the model again (scripts/recheck_claims.py)."""
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher

from pymongo.database import Database

from draft_coach.claims import EXTRACT_VERSION, PASSIVE_MAPS, Claim, check_claim, names_in
from draft_coach.names import NameNormalizer


def save_section(db: Database, section: dict, raw_claims: list[dict], model: str, maps: list[dict],
                 normalizer: NameNormalizer, brawler_info: dict) -> None:
    """Check the raw claims of a section and write its kept claims and its record together."""
    source, records, rejected = section["source"], [], []
    for raw in raw_claims:
        claim = Claim(**raw)
        record, reason = check_claim(claim, section, maps, normalizer, brawler_info)
        if record is None:
            rejected.append({"reason": reason, "concept": claim.concept, "quote": claim.quote})
            continue
        records.append({"_id": f"{section['key']}:{len(records)}", "section_key": section["key"],
                        "source_id": source["id"], "url": f"{source['url']}&t={record['start_s']}s",
                        "version": EXTRACT_VERSION, "model": model, **record})
    db["claims"].delete_many({"section_key": section["key"]})  # a half-saved section is rewritten whole
    if records:
        db["claims"].insert_many(records)
    db["claim_sections"].replace_one({"_id": section["key"]}, {
        "source_id": source["id"], "start_s": section["start_s"], "end_s": section["end_s"],
        "chapter": section["chapter"], "version": EXTRACT_VERSION, "model": model, "raw_claims": raw_claims,
        "kept": len(records), "rejected": rejected, "checked_at": datetime.now(UTC)}, upsert=True)


def print_report(db: Database, normalizer: NameNormalizer, n_sections: int) -> None:
    """End-of-step check: counts of the whole collections, not only of the last run."""
    claims = list(db["claims"].find())
    section_docs = list(db["claim_sections"].find({}, {"rejected": 1, "kept": 1}))
    rejected = Counter(r["reason"] for d in section_docs for r in d["rejected"])
    concept_not_in_sides = sum(1 for c in claims  # a brawler named in the concept but on neither side
                               if set(normalizer.mentions(c["concept"])) - set(c["brawlers"]) - set(c["against_brawlers"]))
    print(f"\n{'sections saved':<28} {len(section_docs)} / {n_sections}")
    print(f"{'claims kept':<28} {len(claims)}")
    print(f"{'claims rejected':<28} {sum(rejected.values())} {dict(rejected)}")
    print(f"{'sections without claims':<28} {sum(1 for d in section_docs if d['kept'] == 0)}")
    print(f"{'claims by topic':<28} {dict(Counter(c['topic'] for c in claims).most_common())}")
    print(f"{'claims with a direction':<28} {sum(1 for c in claims if c['relation'] != 'none')}")
    print(f"{'concept brawler not in sides':<28} {concept_not_in_sides}")
    print(f"{'unknown brawler names':<28} {dict(Counter(n for c in claims for n in c['unknown_brawlers']).most_common(10))}")
    print(f"{'unknown map names':<28} {dict(Counter(n for c in claims for n in c['unknown_maps']).most_common(10))}")
    print(f"{'claims by mode_role':<28} {dict(Counter(r for c in claims for r in c['mode_role']).most_common())}")
    print(f"{'unknown mode_role labels':<28} {dict(Counter(r.lower() for c in claims for r in c['unknown_mode_roles']).most_common(10))}")


def alias_candidates(db: Database, sections_by_key: dict, normalizer: NameNormalizer, brawlers: dict,
                     maps: list[dict]) -> Counter:
    """(brawler id, word of the quote) for claims rejected because the brawler is not in the section: the word of
    the quote most similar to the brawler's name is the likely misspelling ("melody" -> Melodie). Counted, for review."""
    map_names = [m["name"] for m in maps] + [a for m in maps for a in m["aliases"]] + PASSIVE_MAPS
    pairs = Counter()
    for doc in db["claim_sections"].find({"rejected.reason": "brawler_not_in_section"}):
        section = sections_by_key[doc["_id"]]
        seen = ({m for s in section["segments"] for m in s.get("mentions", [])}
                | set(names_in(f"{section['chapter'] or ''} {section['text']}", normalizer, map_names)))
        for r in doc["rejected"]:
            if r["reason"] != "brawler_not_in_section":
                continue
            for brawler in set(normalizer.mentions(r["concept"])) - seen:
                name = brawlers[brawler]["en"].lower()
                words = r["quote"].lower().replace(",", " ").replace(".", " ").split()
                best = max(words, key=lambda w: SequenceMatcher(None, w, name).ratio(), default="")
                close = SequenceMatcher(None, best, name).ratio() >= 0.5  # below: the brawler is not in the quote
                pairs[brawler, best if close else "-"] += 1
    return pairs