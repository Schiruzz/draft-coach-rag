"""Ranked maps: load data/maps.yaml and find a map from any of its spellings."""
from pathlib import Path

import yaml

MODES = {"brawl_ball", "gem_grab", "heist", "hot_zone"}  # aggressive modes (phase 1)
TYPES = {"open", "bushy", "tight", "lane"}               # definitions in dominio-draft.md
FEATURES = {"water_split", "fences", "indestructible_walls", "breakable_key_walls",
            "center_only_path", "flank_bushes", "exposed_objective", "multiple_zones"}
# Features that the Fandom infobox can confirm: the map must have at least one tile of this kind
FEATURE_TILES = {"water_split": "Lake", "fences": "Fence", "indestructible_walls": "IndestructibleWall"}


def load_maps(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def find_map(name: str, maps: list[dict]) -> dict | None:
    """Map whose Fandom name or alias is exactly `name` (case-sensitive, like brawler aliases); None if unknown."""
    return next((m for m in maps if name == m["name"] or name in m["aliases"]), None)