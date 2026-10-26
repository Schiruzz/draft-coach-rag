"""Download the Fandom page of every map in data/maps.yaml and the URL of its picture into data/fandom/maps/
(<id>.wiki and <id>.image_url). The picture itself is not downloaded: the CDN refuses scripts (403).
Files already downloaded are skipped: delete them to download again.
Run: python scripts/fetch_map_pages.py"""
import time
from pathlib import Path

from tqdm import tqdm

from draft_coach.fandom import fetch_page_image, fetch_wikitext
from draft_coach.maps import load_maps

OUT = Path("data/fandom/maps")
PAUSE_S = 0.5  # be gentle with the wiki
OUT.mkdir(parents=True, exist_ok=True)

maps = load_maps(Path("data/maps.yaml"))
missing = []
for m in tqdm(maps, desc="maps"):
    wiki, image_url = OUT / f"{m['id']}.wiki", OUT / f"{m['id']}.image_url"
    try:
        if not wiki.exists():
            wiki.write_text(fetch_wikitext(m["name"]), encoding="utf-8")
        if not image_url.exists():
            image_url.write_text(fetch_page_image(m["name"]), encoding="utf-8")
    except LookupError as error:
        missing.append(str(error))
    time.sleep(PAUSE_S)

# End-of-step check: one page and one picture URL per map
pages = list(OUT.glob("*.wiki"))
urls = list(OUT.glob("*.image_url"))
print(f"\n{'maps in table':<22} {len(maps)}")
print(f"{'pages on disk':<22} {len(pages)}")
print(f"{'picture URLs on disk':<22} {len(urls)}")
print(f"{'not found on Fandom':<22} {missing or 'none'}")
assert len(pages) == len(urls) == len(maps), "some pages or picture URLs are missing: fix the name in maps.yaml"