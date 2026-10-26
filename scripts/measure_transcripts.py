"""List capitalized words that may be misspelled brawler names (step 3q). Read-only.
Run: python scripts/measure_transcripts.py"""
import re
from collections import Counter
from pathlib import Path

from pymongo import MongoClient

from draft_coach.config import MONGO_URI
from draft_coach.names import load_brawlers

CANDIDATES = 60
REPORT = Path("reports/transcript_defects.md")  # git-ignored, like the other reports

segments = [s for doc in MongoClient(MONGO_URI)["draft_coach"]["normalized"].find() for s in doc["segments"]]
brawlers = load_brawlers(Path("data/brawlers.yaml"))
names = {n for b in brawlers for n in (b["en"], b["it"], *b["aliases"], *b["context_aliases"])}
words = Counter(w for s in segments for w in re.findall(r"[A-Za-z][A-Za-z'-]*", s["text"]))
lower = {w for w in words if w[0].islower()}

# A word at the start of a sentence is also found in lower case somewhere else: what is left are proper names,
# among them the names the speech recognizer got wrong ("Goss" for Gus)
suspects = [(w, n) for w, n in words.most_common() if w[0].isupper() and len(w) >= 3 and "'" not in w  # skip "I", "I'm"
            and w.lower() not in lower and w not in names and not any(w.startswith(name) for name in names)]
print(f"{'segments':<16} {len(segments)}")
print(f"{'suspect words':<16} {len(suspects)} distinct, {sum(n for _, n in suspects)} occurrences; the {CANDIDATES} most frequent in the report")
lines = [f"# Suspect names: the {CANDIDATES} most frequent of {len(suspects)}", "", "| word | count | example |", "|---|---|---|"]
for word, count in suspects[:CANDIDATES]:
    example = next(s["text"] for s in segments if re.search(rf"\b{re.escape(word)}\b", s["text"]))
    start = max(example.find(word) - 60, 0)
    lines.append(f"| {word} | {count} | …{example[start:start + 140]}… |")
REPORT.parent.mkdir(exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")
print(f"\nDetails: {REPORT}")