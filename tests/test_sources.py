"""data/sources.yaml must be valid: known types, required fields, unique ids."""
from pathlib import Path

from draft_coach.sources import load_sources


def test_sources_file_is_valid():
    assert load_sources(Path("data/sources.yaml"))