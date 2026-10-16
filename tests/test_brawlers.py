"""Brawler table: unique ids, no name shared by two brawlers, known classes and traits."""
from pathlib import Path

import yaml

BRAWLERS = yaml.safe_load(Path("data/brawlers.yaml").read_text(encoding="utf-8"))
CLASSES = {"thrower", "tank", "space_maker", "anti_tank", "support", "sniper", "control"}  # Bobby's 7 draft classes
TRAITS = {"good_hyper", "knockback_stun", "wall_break", "pierce_damage", "special"}
CLASS_SOURCES = {"bobby-cheat-sheet", "derived"}


def test_ids_are_unique():
    ids = [brawler["id"] for brawler in BRAWLERS]
    assert len(ids) == len(set(ids))


def test_no_name_points_to_two_brawlers():
    owner = {}  # lowercased name -> brawler id
    for brawler in BRAWLERS:
        names = {brawler["en"], brawler["it"], *brawler["aliases"], *brawler["context_aliases"]}
        for name in names:
            assert owner.setdefault(name.lower(), brawler["id"]) == brawler["id"], name


def test_classes_and_traits_are_known():
    for brawler in BRAWLERS:
        assert brawler["class"] in CLASSES, brawler["id"]
        assert set(brawler["traits"]) <= TRAITS, brawler["id"]
        assert brawler["class_source"] in CLASS_SOURCES, brawler["id"]