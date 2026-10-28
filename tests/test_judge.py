"""Retrieval judge helpers: quote check, final verdict, brawler facts, stable ids (no model calls)."""
from pathlib import Path

from draft_coach.judge import (Judgment, PromptBuilder, brawler_facts, build_prompt, final_verdict, judgment_id,
                                quote_found)
from draft_coach.maps import load_maps
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


def test_judgment_id_changes_with_model_and_prompt():
    prompt = build_prompt("Q?", None, "none", "REF", EXCERPT, "none")
    assert judgment_id("gemini-3.5-flash-lite", prompt) == judgment_id("gemini-3.5-flash-lite", prompt)
    assert judgment_id("gemini-3.5-flash-lite", prompt) != judgment_id("gemini-3.8-flash", prompt)
    assert judgment_id("gemini-3.5-flash-lite", prompt) != judgment_id("gemini-3.5-flash-lite", prompt + " ")


def test_prompt_builder_question_facts():
    builder = PromptBuilder(list(BRAWLERS.values()), load_maps(Path("data/maps.yaml")), {})
    check = {"id": "c1", "type": "check", "reference": "REF", "candidate": "willow",
             "scenario": {"quiz_mode": "mythic_plus", "map": "center-stage", "position": 6, "bans": [],
                          "ours": ["crow"], "theirs": ["bibi"]}}
    prompt = builder.prompt(check, EXCERPT, [])
    assert "I want to pick Willow" in prompt and "Mythic+: full draft" in prompt
    assert "Brawlers in the question: Crow (it: Corvo): anti_tank (draft); Bibi" in prompt and "Willow" in prompt
    written = {"id": "w1", "type": "concept", "question": "Why is Bull a space maker?", "reference": "REF"}
    assert "Brawlers in the question: Bull: space_maker (draft)" in builder.prompt(written, EXCERPT, [])


def test_prompt_has_all_parts():
    prompt = build_prompt("Q?", "diamond", "Tick: thrower (draft)", "REF", EXCERPT, "Frank: tank (draft)")
    for part in ["Q?", "Diamond: blind picks", "Tick: thrower", "REF", EXCERPT, "Frank: tank", "Examples"]:
        assert part in prompt