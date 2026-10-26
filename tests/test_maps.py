"""Map table: unique ids and spellings, known modes, every map chapter of the s54 cheat sheet is a known map."""
from pathlib import Path

from draft_coach.maps import FEATURES, MODES, TYPES, find_map, load_maps
from draft_coach.sources import load_sources

MAPS = load_maps(Path("data/maps.yaml"))
CHEAT_SHEET = "spenlc-ranked-cheat-sheet-s54"  # chapter titles are "<Mode> - <Map>"


def test_ids_and_spellings_are_unique():
    ids = [m["id"] for m in MAPS]
    spellings = [name for m in MAPS for name in (m["name"], *m["aliases"])]
    assert len(ids) == len(set(ids))
    assert len(spellings) == len(set(spellings))


def test_fields_are_valid():
    for m in MAPS:
        assert m["mode"] in MODES, m["id"]
        assert isinstance(m["in_pool"], bool), m["id"]
        assert 1 <= len(m["types"]) <= 2 and set(m["types"]) <= TYPES, m["id"]
        assert set(m["features"]) <= FEATURES, m["id"]
        assert isinstance(m.get("notes", ""), str), m["id"]


def test_cheat_sheet_chapters_match_maps():
    source = next(s for s in load_sources(Path("data/sources.yaml")) if s["id"] == CHEAT_SHEET)
    for chapter in source["chapters"]:
        if " - " not in chapter["title"]:
            continue  # intro
        mode, name = chapter["title"].split(" - ", 1)
        mode = mode.lower().replace(" ", "_")
        if mode not in MODES:
            continue  # passive modes: phase 2
        found = find_map(name, MAPS)
        assert found is not None, chapter["title"]
        assert found["mode"] == mode, chapter["title"]