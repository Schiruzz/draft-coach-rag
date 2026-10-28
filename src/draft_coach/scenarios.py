"""Turn a structured draft scenario (eval set, later the quiz) into a plain English question."""
from draft_coach.screens import MODE_NAMES

ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}


def scenario_question(scenario: dict, names: dict, maps: dict, candidate: str | None = None) -> str:
    """names: brawler id -> English name; maps: map id -> map row. A candidate makes it a check question."""
    m = maps[scenario["map"]]

    def listed(ids: list[str], empty: str) -> str:
        return ", ".join(names[i] for i in ids) or empty

    if scenario["quiz_mode"] == "diamond":  # blind picks: no position, enemy picks not visible
        parts = [f"{m['name']} ({MODE_NAMES[m['mode']]}), Diamond, blind pick.",
                 f"Bans: {listed(scenario.get('bans', []), 'not stated')}.",
                 f"My team: {listed(scenario.get('ours', []), 'no picks yet')}."]
    else:
        parts = [f"{m['name']} ({MODE_NAMES[m['mode']]}), {ORDINALS[scenario['position']]} pick.",
                 f"Bans: {listed(scenario.get('bans', []), 'not stated')}.",  # sources often reason without bans
                 f"My team: {listed(scenario.get('ours', []), 'no picks yet')}.",
                 f"Enemy team: {listed(scenario.get('theirs', []), 'no picks yet')}."]
    parts.append(f"I want to pick {names[candidate]}. Is it a good pick?" if candidate else "What should I pick?")
    return " ".join(parts)