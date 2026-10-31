"""Italian game terms -> English terms of the corpus (step 7d)."""
from pathlib import Path

import pytest
import yaml

from draft_coach.glossary import brawler_terms, load_glossary, to_english
from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers

TERMS = {"Arraffa gemme": "Gem Grab", "Rapina": "Heist", "Zona calda": "Hot Zone", "zona calda": "Hot Zone",
         "K.O.": "Knockout", "Dominio": "Hot Zone", "countero": "counter", "Stecca": "Rico", "Ambra": "Amber"}


def test_written_forms_and_multiword():
    assert to_english("Perché perdo sempre in zona calda?", TERMS) == "Perché perdo sempre in Hot Zone?"
    assert to_english("Come countero Bolt in Rapina?", TERMS) == "Come counter Bolt in Heist?"
    assert to_english("Stecca o Ambra in K.O.?", TERMS) == "Rico o Amber in Knockout?"


def test_whole_words_only():
    assert to_english("all'Arraffa gemme", TERMS) == "all'Gem Grab"  # apostrophe is a word boundary
    assert to_english("un Rapinatore", TERMS) == "un Rapinatore"  # not inside a longer word


def test_common_words_match_only_as_written():
    assert to_english("il dominio della mappa", TERMS) == "il dominio della mappa"
    assert to_english("ho preso una stecca", TERMS) == "ho preso una stecca"


def test_conflicting_forms_are_refused(tmp_path):
    path = tmp_path / "glossary.yaml"
    path.write_text("modes:\n  Heist: [Rapina]\n  Bounty: [Rapina]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_glossary(path)


def test_brawler_terms_only_where_names_differ():
    brawlers = [{"en": "Rico", "it": "Stecca"}, {"en": "Bolt", "it": "Bolt"}]
    assert brawler_terms(brawlers) == {"Stecca": "Rico"}


def test_project_glossary_covers_every_map_in_the_pool():
    sections = yaml.safe_load(Path("data/glossary.yaml").read_text(encoding="utf-8"))
    assert set(sections["maps"]) == {m["name"] for m in load_maps(Path("data/maps.yaml"))}
    glossary = load_glossary(Path("data/glossary.yaml"))  # no conflicts
    names = brawler_terms(load_brawlers(Path("data/brawlers.yaml")))
    assert not set(glossary) & set(names)  # an Italian form is a game term or a brawler, never both