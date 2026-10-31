"""Brawler names: load the table, fix known misspellings, find mentions and ambiguous names."""
import re
from pathlib import Path

import yaml


def load_brawlers(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

CORRECTION_WINDOW_S = 20  # around the time of a correction: our times are quote starts, the word may come later


def to_seconds(at: str) -> int:
    """"36:23" -> 2183; long videos keep counting minutes ("127:16")."""
    minutes, seconds = at.split(":")
    return int(minutes) * 60 + int(seconds)


def apply_corrections(source_id: str, segments: list[dict], corrections: list[dict]) -> list[dict]:
    """Point corrections of a raw transcript (data/corrections.yaml), applied before the aliases: the wrong text is
    copied from the raw transcript. Each correction must match exactly once, as whole words and with its case, in the
    segments that start within CORRECTION_WINDOW_S of its time; otherwise ValueError: a correction never fails silently."""
    segments = [dict(s) for s in segments]
    for c in (c for c in corrections if c["source"] == source_id):
        pattern = re.compile(rf"(?<!\w){re.escape(c['wrong'])}(?!\w)")
        hits = [(i, m) for i, s in enumerate(segments) if abs(s["start_s"] - to_seconds(c["at"])) <= CORRECTION_WINDOW_S
                for m in pattern.finditer(s["text"])]
        if len(hits) != 1:
            raise ValueError(f"correction {source_id} {c['at']} {c['wrong']!r}: {len(hits)} matches, exactly 1 needed")
        i, m = hits[0]
        segments[i]["text"] = segments[i]["text"][:m.start()] + c["right"] + segments[i]["text"][m.end():]
    return segments

def word_pattern(names) -> re.Pattern:
    """Match any of the names as a whole word, case-sensitive, longest name first."""
    alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    # (?<!\w) and (?!\w) instead of \b: names like "Mr. P" or "8-Bit" start or end with non-letters
    return re.compile(rf"(?<!\w)({alternatives})(?!\w)")


class NameNormalizer:
    def __init__(self, brawlers: list[dict]):
        self.official = {name: b["id"] for b in brawlers for name in (b["en"], b["it"])}
        self.alias_to_en = {alias: b["en"] for b in brawlers for alias in b["aliases"]}
        self.context_to_ids = {}  # a context alias can belong to several brawlers ("Ali": Alli or Ollie)
        for b in brawlers:
            for alias in b["context_aliases"]:
                self.context_to_ids.setdefault(alias, []).append(b["id"])
        # Official names are in the same pattern so "Larry & Lawrie" is matched whole, not as alias "Larry"
        self.alias_pattern = word_pattern(self.alias_to_en.keys() | self.official.keys())
        self.official_pattern = word_pattern(self.official)
        # Official names in the pattern too, so "Primo" inside "El Primo" is matched as the official name, not as an alias
        self.context_pattern = word_pattern(self.context_to_ids.keys() | self.official.keys())

    def normalize(self, text: str) -> tuple[str, list[str]]:
        """Replace safe aliases with the English name; return the new text and the aliases replaced."""
        replaced = [name for name in self.alias_pattern.findall(text) if name in self.alias_to_en]
        text = self.alias_pattern.sub(lambda m: self.alias_to_en.get(m.group(1), m.group(1)), text)
        return text, replaced

    def mentions(self, text: str) -> list[str]:
        """Brawler ids named in the text with their official EN or IT name, in order of first mention."""
        return list(dict.fromkeys(self.official[name] for name in self.official_pattern.findall(text)))

    def context_candidates(self, text: str) -> list[str]:
        """Ambiguous names left for step 2c (never part of an official name)."""
        return [name for name in self.context_pattern.findall(text) if name in self.context_to_ids]

    def resolve(self, text: str, alias: str, en: str) -> str:
        """Replace a resolved context alias with the English name, leaving official names untouched."""
        pattern = word_pattern({alias} | self.official.keys())
        return pattern.sub(lambda m: en if m.group(1) == alias else m.group(1), text)