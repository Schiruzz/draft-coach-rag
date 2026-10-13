"""Read and validate data/sources.yaml, the list of corpus sources."""
from pathlib import Path

import yaml

REQUIRED = {"id", "type", "title", "creator", "url", "language", "license"}
REQUIRED_BY_TYPE = {"transcript": {"published", "season"}, "wiki": set()}


def load_sources(path: Path) -> list[dict]:
    """Return the listed sources; raise on unknown types, missing fields or duplicate ids."""
    sources = yaml.safe_load(path.read_text(encoding="utf-8"))
    for source in sources:
        if source.get("type") not in REQUIRED_BY_TYPE:
            raise ValueError(f"{source.get('id')}: unknown type {source.get('type')!r}")
        missing = (REQUIRED | REQUIRED_BY_TYPE[source["type"]]) - source.keys()
        if missing:
            raise ValueError(f"{source['id']}: missing fields {sorted(missing)}")
    ids = [source["id"] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source ids in sources.yaml")
    return sources