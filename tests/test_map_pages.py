"""Map page parser on a page in the format of the wiki (from Hot Potato, shortened)."""
import pytest

from draft_coach.map_pages import parse_map_page

PAGE = """{{Protection}}{{MagicWords}}
{{Map Infobox|Creator=AlexEsteAdevarat|Event=Heist|Environment=Brawlywood|Block1=40|Bush=128|IndestructibleWall=20}}
Hot Potato (originally called Bone Tunnel) is a community map for the [[Heist]] event.

==Layout==
The map features a diagonal patch of [[bushes]]. The map is diagonally symmetrical.

==Tips==
*[[Jessie]] can place her turret in the very center of the map.

==History==
*On 27/09/18, Bone Tunnel was added to the game.
*On 11/04/20, Hot Potato was featured in the April Brawl Stars Championship Challenge.
*On 18/09/19, the environment of Hot Potato was changed from the Beach theme to the Old Town theme.
"""


def test_infobox_layout_and_history():
    page = parse_map_page(PAGE)
    assert page["event"] == "Heist" and page["creator"] == "AlexEsteAdevarat" and page["environment"] == "Brawlywood"
    assert page["tiles"] == {"Block1": 40, "Bush": 128, "IndestructibleWall": 20}
    assert page["other"] == {}
    assert page["layout"] == "The map features a diagonal patch of bushes. The map is diagonally symmetrical."
    assert "Jessie" not in page["layout"]  # Tips are left out
    assert [h["date"] for h in page["history"]] == ["2018-09-27", "2019-09-18", "2020-04-11"]  # sorted
    assert page["history"][0]["text"] == "Bone Tunnel was added to the game."


def test_page_without_infobox_raises():
    with pytest.raises(ValueError):
        parse_map_page("==Layout==\nText.")


def test_misspelled_tile_key_is_merged():
    page = parse_map_page("{{Map Infobox|Event=Heist|IndestructibleWall=4|IndestructableWall=3}}")
    assert page["tiles"] == {"IndestructibleWall": 7}