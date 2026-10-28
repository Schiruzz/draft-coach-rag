"""Chunks of the wiki layer (step 5b): mode pages by section, one Layout per map.
The wiki layer is searched apart from the guides: it never takes one of their 5 places."""
import mwparserfromhell as mw

LAYER = "wiki"
INTRO = "Intro"                  # title given to the text before the first section (the rules of the mode)
MODE_SECTIONS = {INTRO, "Tips"}  # Useful Brawlers, Maps, History, Voice Lines are left out (decided 28/10)
MAX_WORDS = 200                  # Tips are cut between bullets into pieces of at most this size (transcripts: median 219)


def plain(wikitext: str) -> str:
    """Wikitext -> plain text on one line (links keep their label, templates and markup go)."""
    return " ".join(mw.parse(wikitext).strip_code().split())


def page_sections(wikitext: str) -> list[dict]:
    """Level-2 sections of a page as raw wikitext, [{title, wikitext}]; the lead first, titled Intro.
    Level-3 sections before the first level-2 one (e.g. Gem Grab 2v2) belong to no section: not ranked variants."""
    sections = []
    for section in mw.parse(wikitext).get_sections(levels=[2], include_lead=True):
        headings = section.filter_headings(recursive=False)
        if headings and section.nodes[0] is headings[0]:  # a level-2 section starts with its heading
            sections.append({"title": headings[0].title.strip_code().strip(),
                             "wikitext": "".join(str(node) for node in section.nodes[1:])})
        else:
            sections.append({"title": INTRO, "wikitext": str(section)})
    return sections


def tips(wikitext: str) -> list[str]:
    """One plain-text tip per bullet; nested bullets (**) join their parent; tables and other lines are dropped."""
    items = []
    for line in wikitext.splitlines():
        if line.startswith("**") and items:
            items[-1] += " " + plain(line.lstrip("*"))
        elif line.startswith("*"):
            items.append(plain(line.lstrip("*")))
    return [item for item in items if item]


def group_words(items: list[str], max_words: int = MAX_WORDS) -> list[str]:
    """Consecutive items joined into pieces of at most max_words; an item longer than that stays alone."""
    pieces, current = [], []
    for item in items:
        if current and len(" ".join(current + [item]).split()) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(item)
    if current:
        pieces.append(" ".join(current))
    return pieces


def mode_texts(title: str, sections: list[dict]) -> list[tuple[str, str]]:
    """[(section label, text)] of a mode page: the Intro whole, the Tips in pieces.
    The mode name leads every text: the Tips often do not name it."""
    by_title = {s["title"]: s["wikitext"] for s in sections}
    texts = [(INTRO, f"{title}, rules: {plain(by_title[INTRO])}")]
    pieces = group_words(tips(by_title["Tips"]))
    return texts + [(f"Tips {i}", f"{title}, tips: {piece}") for i, piece in enumerate(pieces, 1)]


def layout_text(map_doc: dict) -> str:
    """Layout of a map with its name and mode in front: the Layout says "the map", never its name."""
    mode = map_doc["mode"].replace("_", " ").title()  # hot_zone -> Hot Zone
    return f"{map_doc['name']} ({mode}), layout: {map_doc['layout']}"