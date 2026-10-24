"""Card parser on a small hand-written page in the format of the wiki."""
import datetime as dt

from draft_coach.cards import level_11, parse_card, parse_value

PAGE = """{{Brawler Infobox|Class = Marksman|MovementSpeed = 750 (Normal)<br>975 (with Rocket Laces + Gadget Buffie)
|Health = 3000|AttackRange = 9 (Long)<br>11 (with Rocket No. 4 and Buffie)|Reload = 1.95 seconds (Normal)
|AttackLabel = Damage|Attack = 1160|AttackSpeed = 2700<br>3500 (with Rocket Fuel)
|SuperBullets = 9<br>28 (with Hypercharge)|Super = 1040|Gadget1Name = Rocket Laces|Gadget1 = 800|Gadget1Label = Damage
|Gadget2Name = Rocket Fuel|Hypercharge|HyperchargeMultiplier=45%}}
==Attack: Rockin' Rocket==
{{Quote|Short game text.}}Brock fires a [[rocket]].
==Super: Rocket Rain==
Brock fires 9 rockets.
==Gadgets==
===Rocket Laces===
{{Gadget|1}}Brock jumps.
{{Quote|Buffie quote.}}{{Buffie|Gadget}}
With the Buffie, Brock gains speed.
===Rocket Fuel===
Big rocket.
==Star Powers==
===More Rockets!===
More rockets.
===Rocket No. 4===
A fourth rocket.
==Hypercharge: Rocket Barrage==
Cluster bombs.
==Wasabi Power==
Removed event.
==History==
*24/07/19:
**{{Balance|Content=Old change.|Change=Buff}}
*30/06/26:
**{{Balance|Change=Buff|Content=Reload decreased to 1.95 seconds.}}
**{{Balance|Neutral|A skin was added.}}
*01/09/26:
**{{Balance|Nerf|Level up damage decreased in [[Brawl Arena]].}}
*16/09/26:
**{{Balance|Nerf|Super recharge rate decreased.}}
"""
CARD = parse_card(PAGE, since=dt.date(2025, 1, 28))
ABILITIES = {a["name"]: a for a in CARD["abilities"]}


def test_level_11_formulas():
    assert (level_11(3000, 1), level_11(800, 7), level_11(1134, 9)) == (6000, 1000, 1260)


def test_values_with_band_range_and_typos():
    assert parse_value("9 (Long)", {}) == [{"value": 9, "band": "Long"}]
    assert parse_value("3.5%–35% (min to max damage)", {}) == [{"min": 3.5, "max": 35, "unit": "%", "part": "min to max damage"}]
    assert parse_value("750 (Normal", {}) == [{"value": 750, "band": "Normal"}]  # parenthesis left open on the wiki


def test_requirements_are_typed():
    attack_range = ABILITIES["Rockin' Rocket"]["stats"]["range"]
    assert attack_range[1]["requires"] == [{"star_power": "Rocket No. 4"}, {"buffie": "star"}]
    assert CARD["movement_speed"][1]["requires"] == [{"gadget": "Rocket Laces"}, {"buffie": "gadget"}]
    assert ABILITIES["Rocket Rain"]["stats"]["bullets"][1]["requires"] == [{"hypercharge": "Rocket Barrage"}]


def test_stats_go_to_their_ability_at_level_11():
    assert ABILITIES["Rockin' Rocket"]["stats"]["damage"] == [{"label": "Damage", "base": 1160, "level_11": 2320}]
    assert ABILITIES["Rockin' Rocket"]["stats"]["reload"][0]["value"] == 1.95
    assert ABILITIES["Rocket Laces"]["stats"]["damage"][0]["level_11"] == 1000
    assert CARD["health"] == [{"label": "Health", "base": 3000, "level_11": 6000}]


def test_texts_and_buffie():
    laces = ABILITIES["Rocket Laces"]
    assert laces["text"] == "Brock jumps."
    assert laces["buffie"] == {"type": "gadget", "text": "With the Buffie, Brock gains speed."}
    assert ABILITIES["Rockin' Rocket"]["text"] == "Brock fires a rocket."
    assert "Wasabi" not in str(CARD["abilities"])


def test_history_keeps_recent_ranked_buffs_and_nerfs():
    assert [h["date"] for h in CARD["history"]] == ["2026-06-30", "2026-09-16"]