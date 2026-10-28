"""data/sources.yaml must be valid: known types and formats, required fields, unique ids."""
from pathlib import Path

import pytest
import yaml

from draft_coach.sources import load_sources

BASE = {"id": "x", "type": "wiki", "format": "wiki", "title": "t", "creator": "c", "url": "u",
        "language": "en", "license": "l"}


def test_sources_file_is_valid():
    assert load_sources(Path("data/sources.yaml"))


@pytest.mark.parametrize("broken", [
    {"format": "podcast"},                                       # unknown format
    {"format_changes": [{"start_s": 10, "format": "podcast"}]},  # unknown format inside the video
    {"chapter_titles": "useful"},                                # chapter_titles without chapters
    {"chapters": [{"start_s": 0, "title": "Intro"}]},            # chapters without chapter_titles
])
def test_broken_sources_are_caught(tmp_path, broken):
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump([{**BASE, **broken}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_sources(path)