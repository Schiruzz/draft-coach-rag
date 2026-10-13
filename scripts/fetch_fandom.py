"""Download the raw wikitext of every wiki source into data/fandom/. Run: python scripts/fetch_fandom.py"""
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from draft_coach.sources import load_sources

API = "https://brawlstars.fandom.com/api.php"
HEADERS = {"User-Agent": "draft-coach-rag/0.1 (non-commercial, educational)"}  # wikis ask clients to identify
OUT = Path("data/fandom")
OUT.mkdir(parents=True, exist_ok=True)

for source in load_sources(Path("data/sources.yaml")):
    if source["type"] != "wiki":
        continue
    page = unquote(urlparse(source["url"]).path.rsplit("/", 1)[-1])  # .../wiki/Gem_Grab -> Gem_Grab
    params = {"action": "parse", "page": page, "prop": "wikitext", "format": "json", "formatversion": 2}
    response = requests.get(API, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    wikitext = response.json()["parse"]["wikitext"]  # KeyError 'parse' = page not found
    (OUT / f"{source['id']}.wiki").write_text(wikitext, encoding="utf-8")
    print(f"{source['id']:<22} {len(wikitext):>7} chars")