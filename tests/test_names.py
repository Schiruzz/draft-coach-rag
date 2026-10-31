"""Name normalization: safe aliases are replaced, ambiguous ones only reported."""
from pathlib import Path
import pytest

from draft_coach.names import NameNormalizer, apply_corrections, load_brawlers

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

def test_shared_context_alias_keeps_every_brawler():
    assert sorted(NORMALIZER.context_to_ids["Ali"]) == ["alli", "ollie"]  # order follows the table, not meaningful

    

def test_context_alias_inside_an_official_name_is_ignored():
    assert NORMALIZER.context_candidates("El Primo or Primo") == ["Primo"]   # only the standalone one


def test_resolve_leaves_official_names_alone():
    assert NORMALIZER.resolve("El Primo or Primo", "Primo", "El Primo") == "El Primo or El Primo"



# Point corrections (step 7c): applied on the raw text, exactly once, near their time
RAW = [{"start_s": 1170, "text": "So I like a bit everywhere"}, {"start_s": 1186, "text": "a bite of a bit here"},
       {"start_s": 2000, "text": "a bit later"}]


def fix(wrong: str, right: str = "8-Bit") -> list[dict]:
    return apply_corrections("dp", RAW, [{"source": "dp", "at": "19:46", "wrong": wrong, "right": right, "note": ""}])


def test_correction_is_applied_once_in_the_segment_near_its_time():
    assert [s["text"] for s in fix("like a bit", "like 8-Bit")] == ["So I like 8-Bit everywhere", "a bite of a bit here", "a bit later"]


def test_correction_that_matches_nothing_stops_the_run():
    with pytest.raises(ValueError, match="0 matches"):
        fix("a bat")


def test_correction_that_matches_twice_stops_the_run():
    with pytest.raises(ValueError, match="2 matches"):  # "a bite" is not matched: whole words only
        fix("a bit")


def test_corrections_of_other_sources_are_ignored():
    assert apply_corrections("other", RAW, [{"source": "dp", "at": "19:46", "wrong": "a bit", "right": "8-Bit"}]) == RAW