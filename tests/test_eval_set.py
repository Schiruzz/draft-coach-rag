"""Evaluation set: both files are valid, ids are unique across them, and the checks catch broken questions."""
from pathlib import Path

from draft_coach.eval_set import eval_errors, load_eval, picks_before
from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers
from draft_coach.sources import load_sources

SOURCES = load_sources(Path("data/sources.yaml"))
BRAWLERS = load_brawlers(Path("data/brawlers.yaml"))
MAPS = load_maps(Path("data/maps.yaml"))
QUESTIONS = load_eval(Path("data/eval/v1.yaml")) + load_eval(Path("data/eval/v1_holdout.yaml"))


def errors(questions):
    return eval_errors(questions, SOURCES, BRAWLERS, MAPS)


def test_eval_files_are_valid():
    assert errors(QUESTIONS) == []  # also checks ids are unique across the two files


def test_pick_order():
    assert picks_before(1) == (0, 0)
    assert picks_before(3) == (1, 1)  # pick 2 is ours, pick 1 theirs
    assert picks_before(6) == (2, 3)


def test_broken_questions_are_caught():
    base = {"id": "x", "type": "check", "origin": "new", "candidate": "willow", "verdict": "bad", "reference": "r",
            "gold": [{"source": "bobbybs-draft-guide", "start_s": 10, "end_s": 20}],
            "scenario": {"quiz_mode": "mythic_plus", "map": "center-stage", "position": 6,
                         "bans": [], "ours": ["crow", "otis"], "theirs": ["bibi", "kenji", "stu"]}}
    assert errors([base]) == []
    broken = [
        {**base, "verdict": "ok"},
        {**base, "candidate": "otis"},  # already picked
        {**base, "gold": [{"source": "bobbybs-draft-guide", "start_s": 20, "end_s": 10}]},
        {**base, "scenario": {**base["scenario"], "position": 5}},  # pick count does not match
        {**base, "scenario": {**base["scenario"], "theirs": ["bibi", "kenji", "nobody"]}},
        {**base, "scenario": {**base["scenario"], "quiz_mode": "diamond"}},  # diamond sees no enemy picks
        {**base, "verdicts": "bad"},  # misspelled field
        {"id": "y", "type": "abstain", "lang": "it", "question": "q", "reference": "r", "gold": base["gold"]},
    ]
    for question in broken:
        assert errors([question]), question
    assert errors([base, base]) == ["x: duplicate id"]