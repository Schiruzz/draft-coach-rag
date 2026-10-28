"""Read and validate data/sources.yaml, the list of corpus sources."""
from pathlib import Path

import yaml

REQUIRED = {"id", "type", "format", "title", "creator", "url", "language", "license"}
REQUIRED_BY_TYPE = {"transcript": {"published", "season"}, "wiki": set()}
FORMATS = {"guide", "draft_review", "gameplay", "counters", "tier_list", "map_cheat_sheet", "quiz", "wiki"}
CHAPTER_TITLES = {"useful", "ignore"}

def load_sources(path: Path) -> list[dict]:
    """Return the listed sources; raise on unknown types, missing fields or duplicate ids."""
    sources = yaml.safe_load(path.read_text(encoding="utf-8"))
    for source in sources:
        if source.get("type") not in REQUIRED_BY_TYPE:
            raise ValueError(f"{source.get('id')}: unknown type {source.get('type')!r}")
        missing = (REQUIRED | REQUIRED_BY_TYPE[source["type"]]) - source.keys()
        if missing:
            raise ValueError(f"{source['id']}: missing fields {sorted(missing)}")
        formats = [source["format"]] + [change.get("format") for change in source.get("format_changes", [])]
        if not set(formats) <= FORMATS:
            raise ValueError(f"{source['id']}: unknown format in {formats}")
        if ("chapter_titles" in source) != bool(source.get("chapters")) or \
                source.get("chapter_titles", "useful") not in CHAPTER_TITLES:
            raise ValueError(f"{source['id']}: chapter_titles must be useful or ignore, and only with chapters")
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source ids in sources.yaml")
    return sources



def sources_with_formats(sources: list[dict], formats: set[str]) -> list[str]:
    """Ids of the sources whose main format is in formats (format_changes inside a video are not split yet)."""
    return [source["id"] for source in sources if source["format"] in formats]