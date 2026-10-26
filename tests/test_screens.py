"""On-screen lists: valid ids, one per pool map, and the segments built from them."""
from pathlib import Path

import yaml

from draft_coach.maps import load_maps
from draft_coach.names import load_brawlers
from draft_coach.screens import screen_segments
from draft_coach.sources import load_sources

SOURCE = "spenlc-ranked-cheat-sheet-s54"
SCREENS = yaml.safe_load(Path(f"data/screens/{SOURCE}.yaml").read_text(encoding="utf-8"))
MAPS = load_maps(Path("data/maps.yaml"))
BRAWLERS = load_brawlers(Path("data/brawlers.yaml"))


def test_names_and_maps_are_known():
    ids = {b["id"] for b in BRAWLERS}
    for screen in SCREENS:
        for group in screen["groups"]:
            assert set(group["brawlers"]) <= ids, (screen["map"], group["label"])
    assert sorted(s["map"] for s in SCREENS) == sorted(m["id"] for m in MAPS if m["in_pool"])


def test_segments_start_at_the_map_chapter():
    source = next(s for s in load_sources(Path("data/sources.yaml")) if s["id"] == SOURCE)
    segments = screen_segments(SCREENS, source["chapters"], MAPS, BRAWLERS)
    hot_potato = next(s for s in segments if "Hot Potato" in s["text"])
    assert hot_potato["start_s"] == 1089                      # chapter "Heist - Hot Potato"
    assert "1st Pick: Amber" in hot_potato["text"] and "Draft tips: Safe Damage; Flex/Safe; Flex." in hot_potato["text"]
    assert len(segments) == len(SCREENS)