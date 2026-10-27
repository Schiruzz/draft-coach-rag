"""Evaluation set (data/eval/*.yaml): load a file and list its schema errors (no database needed)."""
from pathlib import Path

import yaml

TYPES = {"pick", "check", "habit", "plan", "concept", "abstain"}
CHAT_TYPES = {"habit", "plan", "concept", "abstain"}  # written questions; pick/check are scenarios
CLASSES = {"thrower", "space_maker", "anti_tank", "control", "sniper", "tank", "support"}
QUIZ_MODES = {"diamond", "mythic_plus"}
TEAM_A, TEAM_B = {1, 4, 5}, {2, 3, 6}  # pick order 1, 2-3, 4-5, 6

FIELDS = {  # allowed fields per type (a misspelled field is an error)
    "pick": {"scenario", "accepted_classes", "origin"},
    "check": {"scenario", "candidate", "verdict", "origin"},
    **{t: {"question", "lang"} for t in CHAT_TYPES},
}
COMMON = {"id", "type", "reference", "gold", "covered"}  # covered: only on questions written from a player need


def load_eval(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def picks_before(position: int) -> tuple[int, int]:
    """Mythic+ pick number -> (our picks, their picks) already made."""
    mine = TEAM_A if position in TEAM_A else TEAM_B
    before = range(1, position)
    ours = sum(p in mine for p in before)
    return ours, len(before) - ours


def scenario_errors(s: dict, brawler_ids: set, pool: set) -> list[str]:
    errors = []
    if s.get("quiz_mode") not in QUIZ_MODES:
        errors.append(f"quiz_mode must be one of {sorted(QUIZ_MODES)}")
    if s.get("map") not in pool:
        errors.append(f"map {s.get('map')!r} is not an in-pool map of maps.yaml")
    named = s.get("bans", []) + s.get("ours", []) + s.get("theirs", [])
    errors += [f"unknown brawler {b!r}" for b in named if b not in brawler_ids]
    if len(named) != len(set(named)):
        errors.append("a brawler appears twice among bans and picks")
    if len(s.get("bans", [])) > 6:
        errors.append("more than 6 bans")
    if s.get("quiz_mode") == "diamond":  # blind picks: nothing to see, no pick order
        if s.get("theirs") or "position" in s:
            errors.append("diamond: theirs must be empty and there is no position")
    elif s.get("quiz_mode") == "mythic_plus":
        position = s.get("position")
        if position not in range(1, 7):
            errors.append("mythic_plus: position must be 1-6")
        elif picks_before(position) != (len(s.get("ours", [])), len(s.get("theirs", []))):
            ours, theirs = picks_before(position)
            errors.append(f"pick {position}: expected {ours} of our picks and {theirs} of theirs before it")
    return errors


def question_errors(q: dict, sources: dict, brawler_ids: set, pool: set) -> list[str]:
    kind = q.get("type")
    if kind not in TYPES:
        return [f"type must be one of {sorted(TYPES)}"]
    errors = [f"unknown field {f!r}" for f in q if f not in COMMON | FIELDS[kind]]
    errors += [f"missing field {f!r}" for f in ({"id", "reference"} | FIELDS[kind]) - set(q)]
    if "covered" in q and not isinstance(q["covered"], bool):
        errors.append("covered must be true or false")
    gold = q.get("gold", [])
    if kind == "abstain" and gold:
        errors.append("an abstain question has no gold")
    if kind != "abstain" and not gold:
        errors.append("at least one gold interval is needed")
    for g in gold:
        source = sources.get(g.get("source"))
        if source is None:
            errors.append(f"gold source {g.get('source')!r} is not in sources.yaml")
        elif source["type"] == "transcript" and not 0 <= g.get("start_s", -1) < g.get("end_s", -1):
            errors.append(f"gold {g['source']}: needs 0 <= start_s < end_s")
    if kind in CHAT_TYPES:
        if q.get("lang") not in {"it", "en"} or not q.get("question"):
            errors.append("chat questions need a question and lang it/en")
    else:
        errors += scenario_errors(q.get("scenario") or {}, brawler_ids, pool)
        if q.get("origin") not in {"source_example", "new"}:
            errors.append("origin must be source_example or new")
    if kind == "pick" and (not q.get("accepted_classes") or not set(q["accepted_classes"]) <= CLASSES):
        errors.append(f"accepted_classes must be a non-empty subset of {sorted(CLASSES)}")
    if kind == "check":
        if q.get("verdict") not in {"good", "bad"}:
            errors.append("verdict must be good or bad")
        s = q.get("scenario") or {}
        if q.get("candidate") not in brawler_ids:
            errors.append(f"unknown candidate {q.get('candidate')!r}")
        elif q["candidate"] in s.get("bans", []) + s.get("ours", []) + s.get("theirs", []):
            errors.append("the candidate is already banned or picked")
    return errors


def eval_errors(questions: list[dict], sources: list[dict], brawlers: list[dict], maps: list[dict]) -> list[str]:
    """Every schema error as '<id>: <error>'; an empty list means the set is valid."""
    by_id = {s["id"]: s for s in sources}
    brawler_ids = {b["id"] for b in brawlers}
    pool = {m["id"] for m in maps if m["in_pool"]}
    ids = [q.get("id") for q in questions]
    errors = [f"{i}: duplicate id" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for q in questions:
        errors += [f"{q.get('id')}: {e}" for e in question_errors(q, by_id, brawler_ids, pool)]
    return errors