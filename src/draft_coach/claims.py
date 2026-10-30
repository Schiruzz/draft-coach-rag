"""Step 6: cut transcripts into sections, ask Gemini for the draft claims of each section, check them in code.
A claim is one draft rule (concept) with its dated example, the quote it comes from and its time interval,
read on every aspect of the game: who (brawlers, classes), why (effects, stats), where (mode, map, role),
when in the draft, for which rank band."""
import hashlib
import re
from typing import Literal

from pydantic import BaseModel

from draft_coach.judge import GLOSSARY
from draft_coach.maps import FEATURES, MODES, TYPES, find_map
from draft_coach.names import NameNormalizer, word_pattern

EXTRACT_VERSION = "extract-v5"  # v5: example of a sentence with two rules; all six ranked modes
ALL_MODES = MODES | {"bounty", "knockout"}  # passive modes as data; the phase-1 quiz filters the aggressive ones
EXCLUDED_FORMATS = {"gameplay", "quiz", "wiki"}  # games are cases (after 7b), the quiz is the spec, the wiki is its own layer
SECTION_WORDS = 900  # about 6 minutes of speech; measured on the sample: 312 sections, median 277 words
# Rank bands with different work (Federico): roster building below Diamond (no bans, no draft: which brawlers to max),
# Diamond (bans, blind picks: safe brawlers that can win alone), Mythic+ (full draft: a pool that covers every matchup)
RANK_BANDS = ["roster_building", "diamond", "mythic_plus"]
# Passive-mode maps of the s54 pool (phase 2, not in maps.yaml): removed before looking for brawler names,
# because "Belle's Rock" is a map, not Belle
PASSIVE_MAPS = ["Dry Season", "Hideout", "Layer Cake", "Shooting Star",
                "Belle's Rock", "Flaring Phoenix", "New Horizons", "Out in the Open"]
CLASSES = ["thrower", "space_maker", "anti_tank", "control", "sniper", "tank", "support"]
DRAFT_PHASES = ["ban", "pick_1", "pick_2_3", "pick_4_5", "pick_6", "composition"]
LANE_ROLES = ["mid", "lane"]  # every mode has a mid and two lanes (dominio-draft.md §13)
# Roles in the mode, in the words of the game: the speaker's labels (mode_role_raw) -> 9 roles (Federico, 29/10).
# Built from the labels of the full extract-v4 run; labels not listed here are counted in the report.
def role_key(label: str) -> str:
    """'Zone Time Rollers' -> 'zone time': lowercase, without a trailing brawler/roller word and plural."""
    words = label.lower().replace("-", " ").split()
    if words and words[-1] in {"brawler", "brawlers", "roller", "rollers"}:
        words = words[:-1]
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 and w not in {"pass", "boss"} else w for w in words)

MODE_ROLES = {
    "pressure": ["pressure", "pressure brawler", "pressure brawlers", "aggro", "aggressive", "aggressive brawler",
                 "steal the gems", "base diver", "assassin pressure brawler", "offense"],
    "damage_dealer": ["damage dealer", "damage dealers", "damage", "high damage brawler", "dps"],
    "flex": ["flex", "in between brawler", "flex control", "damage and control"],
    "defender": ["defender", "defensive brawler", "defensive", "defense", "defend", "defending", "safe defender",
                 "defensive roller"],
    "support": ["support", "healer", "aggressive support", "enabler"],
    "gem_carrier": ["gem carrier", "carry the gems", "gem carry"],
    "safe_damage": ["safe damage", "safe damage brawler", "heist damage"],
    "zone_holder": ["zone time", "zone time brawler", "both zones brawler", "both zone brawlers", "both zone brawler"],
    "goal_scorer": ["goal scorer"],
}
# Labels that are not a role in the mode: a class (already in `classes`), a lane, out of scope, or too vague
NOT_MODE_ROLES = {"anti-tank", "control", "controller", "mid brawler", "blue star", "playmaker", "win condition",
                  "cleanup crew", "cleanup", "filler", "tank counter", "mid", "lane brawlers"}
MODE_ROLE_OF = {label: role for role, labels in MODE_ROLES.items() for label in labels}
ROLE_OF_KEY = {role_key(label): role for label, role in MODE_ROLE_OF.items()}  # variants: plurals, "... brawler"
NOT_ROLE_KEYS = {role_key(label) for label in NOT_MODE_ROLES} | {"comfort pick"}  # a habit, not a role
STATS = ["range", "health", "movement_speed", "damage", "reload", "projectile_speed"]  # numbers of the Fandom cards
# Closed vocabulary of ability effects (dominio-draft.md §12)
EFFECTS = [
    "multi_projectile_burst", "piercing", "lobbed", "targeted_area", "self_radius", "dash_attack", "jump_attack",
    "returning", "wall_bounce", "chain_bounce", "split_projectile", "contact_damage",
    "damage_up_with_distance", "damage_down_with_distance", "percent_health_damage", "damage_over_time", "area_damage",
    "charged_attack", "attack_sequence", "empowered_attack", "conditional_buff",
    "stages", "stockpile", "meter", "enrage",
    "alternate_forms", "losable_form", "timed_transformation", "random_super", "multi_use_super",
    "charge_on_damage_taken", "charge_over_time", "charge_on_move", "charge_enemies_near", "charge_allies_near",
    "charge_projectiles_near", "charge_on_heal", "starts_charged", "charge_on_shield_damage",
    "knockback", "stun", "slow", "pull", "trap", "ammo_steal", "mark", "projectile_block",
    "wall_break", "shoot_through_walls", "pass_through_walls", "jump_over_walls", "water_walk",
    "dash_or_jump", "speed_boost",
    "immunity", "self_heal", "shield",
    "heal_allies", "shield_allies", "buff_allies",
    "spawnable",
    "single_ammo", "extra_ammo", "no_auto_reload",
    "objective_limit",
]


class Claim(BaseModel):
    quote: str  # first: the model picks the words before it rewrites them
    concept: str
    reason: str
    example: str
    topic: Literal["class_concept", "ban", "matchup", "meta", "map_mode", "habit", "roster"]
    # who or what the rule is about (the side you pick, or the brawler the speaker describes)
    brawlers: list[str]
    classes: list[Literal[tuple(CLASSES)]]
    effects: list[Literal[tuple(EFFECTS)]]
    stats: list[Literal[tuple(STATS)]]
    # the other side: who or what that side beats or loses to; direction in `relation`
    relation: Literal["strong_against", "weak_against", "none"]
    against_brawlers: list[str]
    against_classes: list[Literal[tuple(CLASSES)]]
    against_effects: list[Literal[tuple(EFFECTS)]]
    draft_phase: list[Literal[tuple(DRAFT_PHASES)]]
    rank_band: list[Literal[tuple(RANK_BANDS)]]
    modes: list[str]
    maps: list[str]
    map_types: list[Literal[tuple(sorted(TYPES))]]
    map_features: list[Literal[tuple(sorted(FEATURES))]]
    lane_role: list[Literal[tuple(LANE_ROLES)]]
    mode_role: list[str]


PROMPT = """You extract draft advice from a section of an automatic transcript of a Brawl Stars video.
Brawler names may be misspelled by the speech recognizer.

Source: "{title}" by {creator} ({format}), published {published}, season {season}.{chapter}

Glossary:
{glossary}

List every claim in the section. A claim is one piece of draft advice the speaker gives as true:
which class or brawler to pick, ban or avoid, when, against what, on which map or mode, in which role,
or a habit that loses drafts. Skip intros, jokes, sponsors, and narration of what happened in a game
unless the speaker draws a rule from it. An idea the speaker mentions only to reject it is not a claim;
the rule the speaker gives instead is. One rule per claim: split a passage with two rules into two claims,
for example "X beats A but loses to B" is two claims: X strong_against A, and X weak_against B.
Write only what the speaker says: never turn advice about one brawler into a rule about its class.

Fields:
- quote: one continuous passage of the section, copied word for word, that states the rule.
  Never skip words and never use "...": if the rule is spread out, quote the shortest continuous passage.
- concept: the rule in one English sentence (at most 30 words) that makes sense alone. If the speaker
  talks about a brawler, name that brawler with its correct name; if about a class, use the class.
- reason: why the rule holds, in the speaker's terms (mechanic, matchup, map, objective); empty if not said.
- example: the brawlers, map or situation the speaker uses to illustrate the rule; empty if none.
  Brawlers in an example are dated: they show the meta of the video, not the rule.
- topic: class_concept (what a class does, team composition), ban, matchup (who counters whom),
  meta (what is strong this season), map_mode (what a map or mode asks for), habit (a mistake players make),
  roster (which brawlers are worth unlocking and maxing, and which upgrades matter, because they are decisive
  on many maps and modes).
Who the rule is about, and against what. A rule has two sides when it says who beats or loses to whom:
fill the first side with who the rule is about, the other side with the opponents or the mechanics that
side beats or loses to, and the direction in relation. Examples: "Stu and Max beat Bo with their speed" ->
brawlers [Stu, Max], effects [speed_boost], relation strong_against, against_brawlers [Bo];
"Bolt suffers slow, stun and knockback" -> brawlers [Bolt], relation weak_against, against_effects
[slow, stun, knockback]; "jumpers punish throwers" -> effects [dash_or_jump], relation strong_against,
against_classes [thrower]. A rule with one side has relation none and the against fields empty.
- brawlers: the brawlers the rule is about, with their correct names; not the brawlers of the example.
- classes: the draft classes the rule is about, mapped with the glossary; empty if none named.
- effects: the mechanics of that side the speaker gives as the reason, from this list: {effects}.
- stats: the numbers of that side the speaker gives as the reason, from this list: {stats}.
- relation: strong_against, weak_against or none.
- against_brawlers, against_classes, against_effects: the other side, with the same lists; empty if none.
- draft_phase: ban, pick_1, pick_2_3, pick_4_5, pick_6 (last pick) or composition (the whole team),
  only if the speaker says when the rule applies; empty otherwise.
- rank_band: only if the speaker or the chapter title says which players the rule is for:
  roster_building (below Diamond: no bans and no draft, the work is choosing which brawlers to max),
  diamond (bans and blind picks, enemy team not visible), mythic_plus (Mythic and above: full draft).
  Empty otherwise: the audience of these videos is Diamond and above. Never infer it from the draft phase
  or from the word "draft": only a rank the speaker names. "The pros" says who does it, not a rank.
- modes: among {modes}, only those the rule is about; empty if any mode.
- maps: map names the rule is about, written as in this list when they match: {maps}; empty if none.
- map_types: among {map_types}, only if the speaker describes the kind of map; empty otherwise.
- map_features: among {map_features}, only if the speaker names that feature of the map; empty otherwise.
- lane_role: mid or lane, only if the speaker says where the brawler or class plays; empty otherwise.
- mode_role: the role in the mode or in the team, in the speaker's words (for example "gem carrier",
  "defender", "pressure", "flex", "damage dealer"), only if the speaker names one; empty otherwise.

If the section has no draft advice, return an empty list.

Section:
{text}"""


def extractable_segments(source: dict, segments: list[dict]) -> list[dict]:
    """Segments sorted by time, without the parts whose format is excluded (s4lamence: gameplay from 14:05)."""
    changes = [{"start_s": 0, "format": source["format"]}] + source.get("format_changes", [])
    def format_at(start_s: int) -> str:
        return [c["format"] for c in changes if c["start_s"] <= start_s][-1]
    return [s for s in sorted(segments, key=lambda s: s["start_s"]) if format_at(s["start_s"]) not in EXCLUDED_FORMATS]


def make_sections(segments: list[dict], chapters: list[dict], max_words: int = SECTION_WORDS) -> list[dict]:
    """Consecutive segments in pieces of at most max_words, without overlap and never across a chapter.
    segments sorted by start_s; chapters [{start_s, title}] sorted (empty list = one chapter)."""
    bounds = chapters or [{"start_s": 0, "title": None}]
    groups = []  # (chapter title, positions in `segments`): positions, not times (screen lists share a time)
    for k, chapter in enumerate(bounds):
        end = bounds[k + 1]["start_s"] if k + 1 < len(bounds) else None
        part = [i for i, s in enumerate(segments) if s["start_s"] >= chapter["start_s"] and (end is None or s["start_s"] < end)]
        current, words = [], 0
        for i in part:
            n = len(segments[i]["text"].split())
            if current and words + n > max_words:  # a single long segment still gets its own section
                groups.append((chapter["title"], current))
                current, words = [], 0
            current.append(i)
            words += n
        if current:
            groups.append((chapter["title"], current))
    sections = []
    for title, positions in groups:
        part, after = [segments[i] for i in positions], positions[-1] + 1
        sections.append({
            "chapter": title,
            "segments": part,
            "start_s": part[0]["start_s"],
            # end: start of the next segment of the transcript (last section: start of its last segment)
            "end_s": segments[after]["start_s"] if after < len(segments) else part[-1]["start_s"],
            "text": " ".join(s["text"] for s in part),
        })
    return sections


def corpus_sections(normalized: list[dict], sources: dict) -> list[dict]:
    """Every section of the corpus, each with its source: normalized transcripts of the formats we extract from."""
    sections = []
    for doc in normalized:
        source = sources[doc["_id"]]
        if source["format"] in EXCLUDED_FORMATS:
            continue
        for section in make_sections(extractable_segments(source, doc["segments"]), source.get("chapters", [])):
            sections.append({**section, "source": source})
    return sections


def build_prompt(source: dict, section: dict, maps: list[dict]) -> str:
    chapter = f'\nChapter: "{section["chapter"]}".' if section["chapter"] and source.get("chapter_titles") == "useful" else ""
    return PROMPT.format(title=source["title"], creator=source["creator"], format=source["format"],
                         published=source["published"], season=source["season"], chapter=chapter,
                         glossary=GLOSSARY, effects=", ".join(EFFECTS), stats=", ".join(STATS),
                         modes=", ".join(sorted(ALL_MODES)), maps=", ".join(m["name"] for m in maps),
                         map_types=", ".join(sorted(TYPES)), map_features=", ".join(sorted(FEATURES)),
                         text=section["text"])


def section_hash(prompt: str, model: str) -> str:
    """Same model and same full prompt (text, source data, version of the instructions) -> same key."""
    return hashlib.sha256(f"{EXTRACT_VERSION}|{model}|{prompt}".encode()).hexdigest()


def _words(text: str) -> list[str]:
    """Lowercase words only: quotes are matched ignoring case, punctuation and spacing."""
    return re.findall(r"\w+", text.lower())


def locate_quote(quote: str, section: dict) -> tuple[int, int] | None:
    """(start_s, end_s) of the segments the quote spans; None if the quote is not in the section."""
    target = _words(quote)
    if not target:
        return None
    words, owner = [], []  # every word of the section and the index of its segment
    for i, segment in enumerate(section["segments"]):
        segment_words = _words(segment["text"])
        words += segment_words
        owner += [i] * len(segment_words)
    for k in range(len(words) - len(target) + 1):
        if words[k:k + len(target)] == target:
            first, last = owner[k], owner[k + len(target) - 1]
            end = section["segments"][last + 1]["start_s"] if last + 1 < len(section["segments"]) else section["end_s"]
            return section["segments"][first]["start_s"], end
    return None


def names_in(text: str, normalizer: NameNormalizer, map_names: list[str]) -> list[str]:
    """Brawler ids whose official EN or IT name is in the text in any case ("griff", "JuJu") or plural ("pipers"),
    map names removed.
    Only for text the model says is about brawlers, or to check it: in a transcript alone "bull" or "kit" may be words."""
    if map_names:  # an empty pattern would match everywhere
        text = re.sub("|".join(re.escape(m) for m in map_names), " ", text, flags=re.IGNORECASE)
    # optional plural "s": "Pipers" in the transcript is Piper (possessives already end the word: "Piper's")
    pattern = re.compile(word_pattern(normalizer.official).pattern.replace(")(?!\\w)", ")s?(?!\\w)"), re.IGNORECASE)
    by_lower = {name.lower(): i for name, i in normalizer.official.items()}
    return list(dict.fromkeys(by_lower[m.lower()] for m in pattern.findall(text)))


def check_claim(claim: Claim, section: dict, maps: list[dict], normalizer: NameNormalizer,
                brawler_info: dict) -> tuple[dict | None, str | None]:
    """Guards after the model: (record, None) if the claim is usable, (None, reason) if it is rejected.
    Interval and brawlers come from the transcript, classes of the brawlers from our tables, not from the model.
    brawler_info: id -> {"class": draft class (brawlers.yaml), "wiki_class": official class (Fandom card)}.
    maps: all maps of maps.yaml (map names are resolved against them and removed before looking for brawlers)."""
    located = locate_quote(claim.quote, section)
    if located is None:
        return None, "quote_not_found"
    if not claim.concept.strip():
        return None, "empty_concept"
    if claim.relation == "none" and (claim.against_brawlers or claim.against_classes or claim.against_effects):
        return None, "against_without_relation"
    # Every brawler the model names (concept, example, both sides) must be said in the section or its chapter
    # title, in any case ("griff" in the transcript): the concept may name a brawler the quote calls "she" or that
    # the title names ("Countering Bo compositions"), never one the speaker did not say (Nani for Najia).
    # Misspellings the transcript still has ("melody" for Melodie) are rejected here and become alias candidates.
    map_names = [m["name"] for m in maps] + [a for m in maps for a in m["aliases"]] + PASSIVE_MAPS
    sides = {"brawlers": claim.brawlers, "against_brawlers": claim.against_brawlers}
    ids = {side: names_in(", ".join(names), normalizer, map_names) for side, names in sides.items()}
    mentions = list(dict.fromkeys(  # the model writes official names with their capitals: case-sensitive here
        normalizer.mentions(re.sub("|".join(re.escape(m) for m in map_names), " ", f"{claim.example} {claim.concept}"))
        + ids["brawlers"] + ids["against_brawlers"]))
    in_section = ({m for s in section["segments"] for m in s.get("mentions", [])}
                  | set(names_in(f"{section['chapter'] or ''} {section['text']}", normalizer, map_names)))
    if not set(mentions) <= in_section:
        return None, "brawler_not_in_section"
    start_s, end_s = located
    found = [find_map(name, maps) for name in claim.maps]
    return {
        **claim.model_dump(exclude={"maps", "modes", "brawlers", "against_brawlers", "mode_role", "lane_role"}),
        "modes": [m for m in claim.modes if m in ALL_MODES],
        "mode_role_raw": claim.mode_role,  # the speaker's words, for the answer
        "mode_role": list(dict.fromkeys(ROLE_OF_KEY[role_key(r)] for r in claim.mode_role if role_key(r) in ROLE_OF_KEY)),
        "lane_role": list(dict.fromkeys(claim.lane_role + (["mid"] if any(role_key(r) == "mid"
                                                                            for r in claim.mode_role) else []))),
        "unknown_mode_roles": [r for r in claim.mode_role  # counted in the report: new labels to place
                               if role_key(r) not in ROLE_OF_KEY and role_key(r) not in NOT_ROLE_KEYS],
        "map_ids": [m["id"] for m in found if m],
        "unknown_maps": [name for name, m in zip(claim.maps, found) if not m],  # counted in the report
        "start_s": start_s,
        "end_s": end_s,
        "mentions": mentions,  # every brawler named, examples included: for search filters
        "brawlers": ids["brawlers"],
        "against_brawlers": ids["against_brawlers"],
        # names the code could not match to a brawler of the table: counted in the report
        "unknown_brawlers": [n for names in sides.values() for n in names if not names_in(n, normalizer, [])],
        # who the brawlers of each side are, from our tables: the claim on Colt is found by a question on anti-tanks
        "brawler_classes": _classes(ids["brawlers"], brawler_info),
        "against_brawler_classes": _classes(ids["against_brawlers"], brawler_info),
    }, None


def _classes(ids: list[str], brawler_info: dict) -> dict:
    """{"draft": draft classes (brawlers.yaml), "game": official classes (Fandom cards)} of the brawlers."""
    return {"draft": list(dict.fromkeys(brawler_info[b]["class"] for b in ids)),
            "game": list(dict.fromkeys(brawler_info[b]["wiki_class"] for b in ids if brawler_info[b]["wiki_class"]))}


def parent_chunk(claim: dict, chunks: list[dict]) -> str | None:
    """Id of the chunk of the same source that overlaps the claim interval the most (small-to-big: search the
    claim, read its chunk). chunks: the chunks of the claim's source, with start_s and end_s."""
    def overlap(c: dict) -> int:
        return min(c["end_s"], claim["end_s"]) - max(c["start_s"], claim["start_s"])
    best = max(chunks, key=overlap, default=None)
    return best["_id"] if best and overlap(best) >= 0 else None


def stale_links(stored: dict[str, str | None], parents: dict[str, str | None]) -> dict[str, str | None]:
    """Point id -> its parent chunk now, for the stored points whose parent chunk changed. The parent depends on the
    claim and on the chunking method: a new method changes it without changing the claim text."""
    return {pid: parents[pid] for pid, chunk in stored.items() if pid in parents and parents[pid] != chunk}

def claim_text(claim: dict, variant: str) -> str:
    """Text embedded for a claim: the rule alone, or the rule with its mechanism (the reason)."""
    return claim["concept"] if variant == "concept" else f"{claim['concept']} {claim['reason']}".strip()