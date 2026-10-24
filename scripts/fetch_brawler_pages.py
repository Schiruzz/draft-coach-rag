"""Download the Fandom page of every brawler in data/brawlers.yaml into data/fandom/brawlers/<id>.wiki.
Pages already downloaded are skipped: delete a file (or the folder) to download it again after a patch.
Run: python scripts/fetch_brawler_pages.py"""
import time
from pathlib import Path

from tqdm import tqdm

from draft_coach.fandom import fetch_wikitext
from draft_coach.names import load_brawlers

OUT = Path("data/fandom/brawlers")
PAUSE_S = 0.5  # be gentle with the wiki
OUT.mkdir(parents=True, exist_ok=True)

brawlers = load_brawlers(Path("data/brawlers.yaml"))
todo = [b for b in brawlers if not (OUT / f"{b['id']}.wiki").exists()]
missing = []
for brawler in tqdm(todo, desc="pages"):
    try:
        (OUT / f"{brawler['id']}.wiki").write_text(fetch_wikitext(brawler["en"]), encoding="utf-8")
    except LookupError as error:
        missing.append(str(error))
    time.sleep(PAUSE_S)

# End-of-step check: one non-empty page per brawler, each with an infobox
files = {f.stem: f.read_text(encoding="utf-8") for f in OUT.glob("*.wiki")}
no_infobox = sorted(stem for stem, text in files.items() if "Infobox" not in text)
print(f"\n{'brawlers in table':<22} {len(brawlers)}")
print(f"{'pages on disk':<22} {len(files)}")
print(f"{'not found on Fandom':<22} {missing or 'none'}")
print(f"{'pages without infobox':<22} {no_infobox or 'none'}")
assert len(files) == len(brawlers) and not no_infobox, "some pages are missing or not brawler pages"