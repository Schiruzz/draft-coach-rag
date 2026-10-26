"""Map page from Fandom: infobox (creator, event, environment, tile counts), Layout text, dated History.
Tips are left out on purpose: undated and tied to old metas."""
import datetime as dt
import re

import mwparserfromhell as mw

TEXT_FIELDS = {"Creator": "creator", "Event": "event", "Environment": "environment"}
TILE_TYPOS = {"IndestructableWall": "IndestructibleWall"}  # misspelled key found on one map page
HISTORY_LINE = re.compile(r"^\*\s*On (\d{2})/(\d{2})/(\d{2}),\s*(.*)$")  # "*On 27/09/18, Bone Tunnel was added..."


def section_lines(wikitext: str, title: str) -> list[str]:
    """Lines of a level-2 section, heading excluded; [] if the page has no such section."""
    sections = mw.parse(wikitext).get_sections(matches=title, levels=[2])
    return str(sections[0]).splitlines()[1:] if sections else []


def parse_infobox(wikitext: str) -> dict:
    """{creator, event, environment, tiles: {Bush: 128, ...}, other: {...}}; raise if the page has no map infobox."""
    infobox = next((t for t in mw.parse(wikitext).filter_templates() if t.name.strip() == "Map Infobox"), None)
    if infobox is None:
        raise ValueError("no Map Infobox: not a map page")
    fields = {str(p.name).strip(): p.value.strip_code().strip() for p in infobox.params}
    result = {key: fields.get(name) for name, key in TEXT_FIELDS.items()}
    result["tiles"] = {}
    for k, v in fields.items():
        if k not in TEXT_FIELDS and v.isdigit():
            key = TILE_TYPOS.get(k, k)
            result["tiles"][key] = result["tiles"].get(key, 0) + int(v)
    # Anything else is kept as text, never dropped: the build check prints it
    result["other"] = {k: v for k, v in fields.items() if k not in TEXT_FIELDS and not v.isdigit()}
    return result


def parse_history(wikitext: str) -> list[dict]:
    """[{date: ISO, text}] sorted by date (the wiki list is not always in order)."""
    history = []
    for line in section_lines(wikitext, "History"):
        match = HISTORY_LINE.match(line.strip())
        if match:
            day, month, year, text = match.groups()
            date = dt.date(2000 + int(year), int(month), int(day)).isoformat()
            history.append({"date": date, "text": mw.parse(text).strip_code().strip()})
    return sorted(history, key=lambda h: h["date"])


def parse_map_page(wikitext: str) -> dict:
    layout = mw.parse("\n".join(section_lines(wikitext, "Layout"))).strip_code().strip()
    return {**parse_infobox(wikitext), "layout": layout, "history": parse_history(wikitext)}