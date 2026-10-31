"""Italian game terms in a question -> the English terms of the corpus (step 7d).
The embedding model is multilingual, but it cannot know the names of the Italian game client
(Rapina = Heist, Ambra = Amber): they are replaced before the search."""
import re
from pathlib import Path

import yaml

from draft_coach.names import load_brawlers


def load_glossary(path: Path) -> dict[str, str]:
    """data/glossary.yaml (sections of English term -> Italian forms) -> {Italian form: English term}.
    The same Italian form under two English terms raises ValueError."""
    terms = {}
    for section in yaml.safe_load(path.read_text(encoding="utf-8")).values():
        for english, forms in section.items():
            for form in forms:
                if terms.get(form, english) != english:
                    raise ValueError(f"{form!r} is both {terms[form]!r} and {english!r}")
                terms[form] = english
    return terms


def brawler_terms(brawlers: list[dict]) -> dict[str, str]:
    """Italian brawler names that differ from the English ones (data/brawlers.yaml): {Italian: English}."""
    return {b["it"]: b["en"] for b in brawlers if b["it"] != b["en"]}


def to_english(question: str, terms: dict[str, str]) -> str:
    """Replace whole-word Italian terms with their case, longest first ("Zona calda" before a shorter form).
    Case counts because several terms are common words (Dominio, Ponte, Corvo, Stecca): only the written forms match."""
    if not terms:
        return question
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.sub(rf"(?<!\w)(?:{alternatives})(?!\w)", lambda m: terms[m.group(0)], question)


def load_terms() -> dict[str, str]:
    """All the Italian terms the retrieval replaces: the glossary and the Italian brawler names (repo root paths)."""
    return {**load_glossary(Path("data/glossary.yaml")), **brawler_terms(load_brawlers(Path("data/brawlers.yaml")))}