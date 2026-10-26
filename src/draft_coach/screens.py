"""Lists shown on screen in a video (data/screens/<source id>.yaml) turned into transcript-like segments."""
from draft_coach.maps import find_map

MODE_NAMES = {"brawl_ball": "Brawl Ball", "gem_grab": "Gem Grab", "heist": "Heist", "hot_zone": "Hot Zone"}


def chapter_starts(chapters: list[dict], maps: list[dict]) -> dict:
    """Map id -> start of its chapter; chapter titles are "<Mode> - <Map>" (others, e.g. Intro, are skipped)."""
    starts = {}
    for chapter in chapters:
        if " - " in chapter["title"]:
            found = find_map(chapter["title"].split(" - ", 1)[1], maps)
            if found:
                starts[found["id"]] = chapter["start_s"]
    return starts


def screen_segments(screens: list[dict], chapters: list[dict], maps: list[dict], brawlers: list[dict]) -> list[dict]:
    """One segment per map: [{start_s, text}], with English brawler names so the name pipeline finds them."""
    starts = chapter_starts(chapters, maps)
    by_id = {m["id"]: m for m in maps}
    en = {b["id"]: b["en"] for b in brawlers}
    segments = []
    for screen in screens:
        m = by_id[screen["map"]]
        groups = " ".join(f"{g['label']}: {', '.join(en[b] for b in g['brawlers'])}." for g in screen["groups"])
        text = (f"[On screen] {m['name']} ({MODE_NAMES[m['mode']]}) draft cheat sheet. {groups} "
                f"Draft tips: {'; '.join(screen['composition'])}. {' '.join(t.rstrip('.') + '.' for t in screen['tips'])}")
        segments.append({"start_s": starts[m["id"]], "text": text})
    return segments