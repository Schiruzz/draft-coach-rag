"""Name normalization: safe aliases are replaced, ambiguous ones only reported."""
from pathlib import Path

from draft_coach.names import NameNormalizer, load_brawlers

NORMALIZER = NameNormalizer(load_brawlers(Path("data/brawlers.yaml")))


def test_safe_aliases_become_english_names():
    text, replaced = NORMALIZER.normalize("Klette and Dyna with Mr P")
    assert text == "Colette and Dynamike with Mr. P"
    assert replaced == ["Klette", "Dyna", "Mr P"]


def test_official_name_is_not_replaced_inside_itself():
    assert NORMALIZER.normalize("Larry & Lawrie or Larry")[0] == "Larry & Lawrie or Larry & Lawrie"


def test_alias_inside_a_word_is_ignored():
    assert NORMALIZER.normalize("Dynamic Kazakh")[0] == "Dynamic Kazakh"


def test_mentions_use_en_and_it_names_once():
    assert NORMALIZER.mentions("Colette e Bombardino, poi Colette") == ["colette", "barley"]


def test_context_aliases_are_only_reported():
    text = "give me the cult"
    assert NORMALIZER.normalize(text)[0] == text
    assert NORMALIZER.context_candidates(text) == ["me", "cult"]