"""Wiki layer chunks (step 5b): sections, tips, pieces, texts with the name in front."""
from draft_coach.wiki_chunks import INTRO, group_words, layout_text, mode_texts, page_sections, tips

PAGE = """{{Quote|Collect gems!}}
In the [[Gem Grab]] Event, two teams of three.
===Gem Grab 2v2===
Not ranked.
==Maps==
*[[Hard Rock Mine]]
==Tips==
{| class="article-table"
!Power Level
|-
|1
|}
*Keep the [[Gem Carrier]] safe.
**Even at the end.
*Control the mine.
==History==
*Added.
"""


def test_page_sections_lead_is_intro_and_level3_before_level2_is_dropped():
    sections = page_sections(PAGE)
    assert [s["title"] for s in sections] == [INTRO, "Maps", "Tips", "History"]
    assert "two teams" in sections[0]["wikitext"] and "2v2" not in sections[0]["wikitext"]
    assert not sections[2]["wikitext"].startswith("==")  # the heading is not part of the body


def test_tips_one_per_bullet_nested_joined_tables_dropped():
    body = page_sections(PAGE)[2]["wikitext"]
    assert tips(body) == ["Keep the Gem Carrier safe. Even at the end.", "Control the mine."]


def test_group_words_never_over_the_limit_except_a_long_item():
    assert group_words(["a b", "c d", "e f"], max_words=4) == ["a b c d", "e f"]
    assert group_words(["a b c d e", "f"], max_words=4) == ["a b c d e", "f"]  # a long tip stays whole


def test_mode_texts_name_in_front():
    texts = mode_texts("Gem Grab", page_sections(PAGE))
    assert texts[0] == (INTRO, "Gem Grab, rules: In the Gem Grab Event, two teams of three.")
    assert texts[1] == ("Tips 1", "Gem Grab, tips: Keep the Gem Carrier safe. Even at the end. Control the mine.")


def test_layout_text_name_and_mode_in_front():
    doc = {"name": "Hot Potato", "mode": "heist", "layout": "The map features bushes."}
    assert layout_text(doc) == "Hot Potato (Heist), layout: The map features bushes."
    assert layout_text({**doc, "mode": "hot_zone"}).startswith("Hot Potato (Hot Zone)")