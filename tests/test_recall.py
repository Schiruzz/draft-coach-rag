"""Recall helpers: overlap rule, first hit rank, coverage, sign test, scenario questions."""
from pathlib import Path

from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers
from draft_coach.recall import coverage, first_hit, hits_gold, sign_test_p, wilson_interval
from draft_coach.scenarios import scenario_question

GOLD = {"source": "bobbybs-draft-guide", "start_s": 100, "end_s": 130}
NAMES = {b["id"]: b["en"] for b in load_brawlers(Path("data/brawlers.yaml"))}
MAPS = {m["id"]: m for m in load_maps(Path("data/maps.yaml"))}


def chunk(source, start, end):
    return {"source_id": source, "start_s": start, "end_s": end}


def test_overlap_rule():
    assert hits_gold(chunk("bobbybs-draft-guide", 60, 120), GOLD)       # 20 of 30 s inside
    assert hits_gold(chunk("bobbybs-draft-guide", 115, 175), GOLD)      # exactly half inside
    assert not hits_gold(chunk("bobbybs-draft-guide", 125, 185), GOLD)  # only 5 s inside
    assert not hits_gold(chunk("bief-counters-every-brawler", 100, 130), GOLD)  # other source


def test_first_hit_rank():
    chunks = [chunk("bief-counters-every-brawler", 0, 60), chunk("bobbybs-draft-guide", 90, 150)]
    assert first_hit(chunks, [GOLD]) == 2
    assert first_hit(chunks[:1], [GOLD]) is None


def test_coverage():
    other = {"source": "bief-counters-every-brawler", "start_s": 0, "end_s": 20}
    assert coverage([chunk("bobbybs-draft-guide", 90, 150)], [GOLD, other]) == 0.5


def test_sign_test():
    assert sign_test_p(0, 0) == 1.0
    assert round(sign_test_p(13, 9), 2) == 0.52  # the A/B split of 28/10: not a real difference
    assert sign_test_p(10, 0) < 0.01

def test_wilson_interval():
    low, high = wilson_interval(70, 88)  # about 80% on the 88 written questions: about 8 points each side
    assert round(low, 2) == 0.70 and round(high, 2) == 0.87
    assert wilson_interval(10, 10)[1] == 1.0 and wilson_interval(10, 10)[0] > 0.7  # never above 100%
    assert wilson_interval(0, 0) == (0.0, 1.0)

def test_scenario_question_check():
    scenario = {"quiz_mode": "mythic_plus", "map": "center-stage", "position": 6, "bans": [],
                "ours": ["crow", "otis"], "theirs": ["bibi", "kenji", "stu"]}
    assert scenario_question(scenario, NAMES, MAPS, "willow") == (
        "Center Stage (Brawl Ball), 6th pick. Bans: not stated. My team: Crow, Otis. "
        "Enemy team: Bibi, Kenji, Stu. I want to pick Willow. Is it a good pick?")


def test_scenario_question_diamond_pick():
    scenario = {"quiz_mode": "diamond", "map": "center-stage", "bans": ["shade"], "ours": [], "theirs": []}
    assert scenario_question(scenario, NAMES, MAPS) == (
        "Center Stage (Brawl Ball), Diamond, blind pick. Bans: Shade. My team: no picks yet. What should I pick?")