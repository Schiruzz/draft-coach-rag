"""Retrieval judge helpers: quote check, final verdict, brawler facts, stable ids (no model calls)."""
from pathlib import Path

from draft_coach.judge import Judgment, brawler_facts, build_prompt, final_verdict, judgment_id, quote_found
from draft_coach.names import load_brawlers

BRAWLERS = {b["id"]: b for b in load_brawlers(Path("data/brawlers.yaml"))}
EXCERPT = "If you also said Frank, you are correct because they won't have enough damage to take out another high HP brawler."


def test_quote_ignores_case_and_punctuation():
    assert quote_found("they WON'T have enough damage", EXCERPT)
    assert not quote_found("they have enough damage", EXCERPT)
    assert not quote_found("", EXCERPT)  # an empty quote never proves anything


def test_yes_without_real_quote_becomes_no():
    assert final_verdict(Judgment(reason="r", verdict="yes", quote="Frank is a great pick"), EXCERPT) == "no"
    assert final_verdict(Judgment(reason="r", verdict="partial", quote="If you also said Frank"), EXCERPT) == "partial"
    assert final_verdict(Judgment(reason="r", verdict="no", quote=""), EXCERPT) == "no"


def test_brawler_facts_join_draft_and_game_class():
    facts = brawler_facts(["crow", "bull", "crow"], BRAWLERS, {"bull": "Tank"})
    assert facts == "Crow (it: Corvo): anti_tank (draft); Bull: space_maker (draft), Tank (game)"
    assert brawler_facts([], BRAWLERS, {}) == "none"


def test_judgment_id_changes_with_reference():
    assert judgment_id("q1", "ref", EXCERPT) == judgment_id("q1", "ref", EXCERPT)
    assert judgment_id("q1", "ref", EXCERPT) != judgment_id("q1", "new ref", EXCERPT)


def test_prompt_has_all_parts():
    prompt = build_prompt("Q?", "diamond", "Tick: thrower (draft)", "REF", EXCERPT, "Frank: tank (draft)")
    for part in ["Q?", "Diamond: blind picks", "Tick: thrower", "REF", EXCERPT, "Frank: tank", "Examples"]:
        assert part in prompt