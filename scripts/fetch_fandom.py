"""Download the raw wikitext of every wiki source into data/fandom/. Run: python scripts/fetch_fandom.py"""
from pathlib import Path
from urllib.parse import unquote, urlparse

from draft_coach.fandom import fetch_wikitext
from draft_coach.sources import load_sources

OUT = Path("data/fandom")
OUT.mkdir(parents=True, exist_ok=True)

for source in load_sources(Path("data/sources.yaml")):
    if source["type"] != "wiki":
        continue
    page = unquote(urlparse(source["url"]).path.rsplit("/", 1)[-1])  # .../wiki/Gem_Grab -> Gem_Grab
    wikitext = fetch_wikitext(page)
    (OUT / f"{source['id']}.wiki").write_text(wikitext, encoding="utf-8")
    print(f"{source['id']:<22} {len(wikitext):>7} chars")