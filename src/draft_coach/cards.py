"""Brawler card from a Fandom page: infobox numbers at level 11, ability texts with buffies, recent balance history."""
import datetime as dt
import difflib
import math
import re

import mwparserfromhell as mw

BANDS = {"Very Short", "Short", "Normal", "Long", "Very Long", "Very Slow", "Slow", "Fast", "Very Fast"}
# Infobox prefix -> (ability key, level at which the wiki stores the value)
PREFIXES = [("Attack", "attack", 1), ("Super", "super", 1), ("Gadget1", "gadget1", 7), ("Gadget2", "gadget2", 7),
            ("StarPower1", "star_power1", 9), ("StarPower2", "star_power2", 9), ("Hypercharge", "hypercharge", None)]
NOT_STATS = {"Title", "PrestigeTitle", "Rarity", "Class", "VoiceActor", "Previous", "Next", "Image", "Name", "GadgetCharges"}
NUMBER = re.compile(r"^([+-]?\d[\d.,]*)(?:\s*%?\s*[–-]\s*(\d[\d.,]*))?\s*(%|°|º|seconds?|tiles?)?\s*(?:\((.*?)\)?)?\s*$")
NOT_RANKED = re.compile(r"Wasabi Power|Mutation|Brawl Arena")  # temporary events and other modes: not ranked balance
# Kaze's own infobox shows these values at level "infinity": they do not grow with the level
FIXED = {("Kaze Infobox", "gadget1"), ("Kaze Infobox", "star_power1")}
BUFFIE_TYPES = {"gadget": "gadget", "star": "star", "hyper": "hyper"}


def level_11(base: float, stored_level: int) -> int:
    """Value at level 11 with the formulas of the wiki templates (common one verified on 8 brawlers, 48/48 values)."""
    if stored_level == 0:   # fixed value (see FIXED)
        return math.floor(base)
    if stored_level == 1:   # health, attack, super: +10% of the level-1 value per level
        return int(base + math.floor(base / 10) * 10)
    if stored_level == 7:   # gadgets: value stored at level 7
        return math.floor(base / 16 * 20)
    return math.floor(base / 18 * 20)  # star powers: value stored at level 9


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def to_number(text: str) -> int | float:
    value = float(text.replace(",", ""))
    return int(value) if value.is_integer() else value


def parse_value(raw: str, abilities: dict) -> list[dict]:
    """'9 (Long)<br>11 (with Rocket No. 4 and Buffie)' -> [{value: 9, band: Long}, {value: 11, requires: [...]}].
    A line that is not a number is kept as {"raw": text}: nothing is dropped, nothing is guessed."""
    items = []
    for line in re.split(r"<br\s*/?>", raw):
        line = mw.parse(line).strip_code().strip()
        if not line:
            continue
        match = NUMBER.match(line)
        if not match:
            items.append({"raw": line})
            continue
        low, high, unit, notes = match.groups()
        item = {"value": to_number(low)} if high is None else {"min": to_number(low), "max": to_number(high)}
        if unit:
            item["unit"] = {"º": "°", "second": "seconds", "tile": "tiles"}.get(unit, unit)
        for part in [p.strip() for p in (notes or "").split(";") if p.strip()]:
            condition = re.match(r"^(?:with|while)\s+(.*)$", part)
            if condition:
                item["requires"] = parse_requires(condition.group(1), abilities)
            elif part in BANDS:
                item["band"] = part
            else:
                item["part"] = part  # e.g. "jar", "snake", "fire", "Power Level 9–11"
        items.append(item)
    return items


def parse_requires(text: str, abilities: dict) -> list[dict]:
    """'Rocket No. 4 and Buffie' -> [{star_power: Rocket No. 4}, {buffie: star}].
    Types: an ability of the page, buffie, gear, or state (anything else, as the wiki writes it)."""
    out, last_type = [], None
    hypercharge = next((name for kind, name in abilities.values() if kind == "hypercharge"), "Hypercharge")
    for name in re.split(r"\s+and\s+|\s*\+\s*", text.strip()):
        name = re.sub(r"^(?:with|while)\s+|\s+is active$", "", name.strip())  # "while X is active" -> "X"
        key = norm(name)
        if "buffie" in key:
            kind = next((t for word, t in BUFFIE_TYPES.items() if key.startswith(word)), None)
            kind = kind or {"gadget": "gadget", "star_power": "star", "hypercharge": "hyper"}.get(last_type)
            out.append({"buffie": kind or "unknown"})
        elif key.startswith("hyperch"):  # "Hypercharge", "Hypercharged Super", typo "Hyperchange"
            out.append({"hypercharge": hypercharge})
            last_type = "hypercharge"
        elif key.endswith("gear"):
            out.append({"gear": name})
        elif key in abilities or (close := difflib.get_close_matches(key, list(abilities), n=1, cutoff=0.85)):
            ability_type, ability_name = abilities[key if key in abilities else close[0]]
            out.append({ability_type: ability_name})
            last_type = ability_type
        else:
            out.append({"state": name})  # e.g. "Super", "full Rage", "three monster notes"
    return out


def plain(wikicode) -> str:
    """Wikitext -> readable text (templates such as quotes and icons are dropped, links keep their text)."""
    return re.sub(r"\s+", " ", mw.parse(str(wikicode)).strip_code()).strip()


def split_buffie(body: str) -> tuple[str, dict | None]:
    """Ability text and its buffie ({{Buffie|Gadget}} marks where the buffie text starts)."""
    match = re.search(r"\{\{\s*Buffie\s*\|\s*(\w+)\s*\}\}", body)
    if not match:
        return plain(body), None
    kind = BUFFIE_TYPES.get(match.group(1).lower(), match.group(1).lower())
    return plain(body[:match.start()]), {"type": kind, "text": plain(body[match.end():])}


def ability_sections(code) -> tuple[list[dict], list[dict]]:
    """Abilities and traits from the level-2 sections of the page."""
    abilities, traits = [], []
    for section in code.get_sections(levels=[2]):
        title = str(section.filter_headings()[0].title).strip()
        head, _, name = title.partition(":")
        subs = section.get_sections(levels=[3])
        if head in ("Attack", "Super", "Hypercharge"):
            text, buffie = split_buffie(str(section).split("\n", 1)[1] if "\n" in str(section) else "")
            abilities.append({"type": head.lower(), "name": name.strip(), "text": text, "buffie": buffie})
        elif head in ("Attacks", "Supers"):  # one ability per form, e.g. "Cannon: Loose Tooth"
            for sub in subs:
                sub_title = str(sub.filter_headings()[0].title).strip()
                form, _, sub_name = sub_title.partition(":") if ":" in sub_title else ("", "", sub_title)
                text, buffie = split_buffie(str(sub).split("\n", 1)[1])
                abilities.append({"type": head[:-1].lower(), "name": sub_name.strip(), "form": form.strip() or None,
                                  "text": text, "buffie": buffie})
        elif head in ("Gadgets", "Star Powers"):
            for sub in subs:
                text, buffie = split_buffie(str(sub).split("\n", 1)[1] if "\n" in str(sub) else "")
                abilities.append({"type": "gadget" if head == "Gadgets" else "star_power",
                                  "name": str(sub.filter_headings()[0].title).strip(), "text": text, "buffie": buffie})
        elif head in ("Trait", "Traits", "Ultra Trait", "Rage Bar", "Mecha"):
            if subs and head == "Traits":
                traits += [{"name": str(s.filter_headings()[0].title).strip(), "text": plain(str(s).split("\n", 1)[1])} for s in subs]
            else:
                traits.append({"name": head, "text": plain(str(section).split("\n", 1)[1])})
    return abilities, traits


def history(code, since: dt.date) -> list[dict]:
    """Buffs and nerfs dated on or after `since` (neutral changes such as skins are left out)."""
    sections = [s for s in code.get_sections(levels=[2]) if str(s.filter_headings()[0].title).strip() == "History"]
    out, date = [], None
    for line in str(sections[0]).splitlines() if sections else []:
        found = re.match(r"^\*\s*(?:On )?(\d{2})/(\d{2})/(\d{2})", line)
        if found:
            day, month, year = map(int, found.groups())
            date = dt.date(2000 + year, month, day)
            continue
        if not (line.startswith("**") and date and date >= since):
            continue
        for template in mw.parse(line).filter_templates():
            if str(template.name).strip() != "Balance":
                continue
            params = {str(p.name).strip(): str(p.value).strip() for p in template.params}
            change = (params.get("Change") or params.get("1", "")).lower()
            text = params.get("Content") or params.get("2") or params.get("1", "")
            if change in ("buff", "nerf") and not NOT_RANKED.search(text):
                out.append({"date": date.isoformat(), "change": change, "text": plain(text)})
    return out


def parse_card(wikitext: str, since: dt.date) -> dict:
    code = mw.parse(wikitext)
    box = next(t for t in code.filter_templates(recursive=False) if str(t.name).strip().replace("_", " ").endswith("Infobox"))
    template = str(box.name).strip().replace("_", " ")
    params = {str(p.name).strip(): str(p.value).strip() for p in box.params if str(p.value).strip()}
    abilities, traits = ability_sections(code)
    by_name = {norm(a["name"]): (a["type"], a["name"]) for a in abilities if a["name"]}

    # Infobox fields grouped by ability; damage-like values (Attack, Attack2, Gadget12...) get their level-11 value
    stats = {key: {} for _, key, _ in PREFIXES}
    health, top = [], {}
    for field, raw in params.items():
        if field in NOT_STATS or field.isdigit() or "Label" in field or field.endswith("Name"):
            continue
        if re.fullmatch(r"Health\d?", field):  # Kaze: Health1, Health2 (one per form)
            index = field[6:]
            base = to_number(raw.split()[0])
            health.append({"label": params.get(f"HealthLabel{index}", "Health"), "base": base, "level_11": level_11(base, 1)})
            continue
        prefix = next(((p, k, lvl) for p, k, lvl in PREFIXES if field.startswith(p)), None)
        if prefix is None:  # MovementSpeed, Reload (attack reload without prefix; Kaze: Reload1, Reload2)
            if re.fullmatch(r"Reload\d?", field):
                stats["attack"][field.lower()] = parse_value(raw, by_name)
            else:
                top[field] = parse_value(raw, by_name)
            continue
        p, key, stored = prefix
        rest = field[len(p):]
        if (template, key) in FIXED:
            stored = 0
        if stored is not None and re.fullmatch(r"\d{0,2}", rest):  # Attack, Attack2, Gadget12; Kaze: Attack12
            value = parse_value(raw, by_name)[0]
            base = value.get("value", value.get("max"))
            label = params.get(f"{p}Label{rest}") or params.get(f"{p}Label{rest or '1'}") or "Damage"
            stats[key].setdefault("damage", []).append({"label": label, "base": base, "level_11": level_11(base, stored)})
        else:
            stats[key][re.sub(r"(?<!^)(?=[A-Z])", "_", rest).lower() or "value"] = parse_value(raw, by_name)

    # Attach each group of stats to its ability: gadget 1/2 by the name in the infobox, otherwise by order
    def find(kind, number=1, name=None):
        same = [a for a in abilities if a["type"] == kind]
        named = [a for a in same if name and norm(a["name"]) == norm(name)]
        return (named or same[number - 1:number] or [None])[0]
    targets = {"attack": find("attack"), "super": find("super"), "hypercharge": find("hypercharge"),
               "gadget1": find("gadget", 1, params.get("Gadget1Name")), "gadget2": find("gadget", 2, params.get("Gadget2Name")),
               "star_power1": find("star_power", 1, params.get("StarPower1Name")),
               "star_power2": find("star_power", 2, params.get("StarPower2Name"))}
    for key, ability in targets.items():
        if ability is not None and stats[key]:
            ability["stats"] = stats[key]
    return {"wiki_class": params.get("Class"), "rarity": params.get("Rarity"), "health": health,
            "movement_speed": top.pop("MovementSpeed", []), "other_stats": top, "trait": traits,
            "abilities": abilities, "history": history(code, since)}