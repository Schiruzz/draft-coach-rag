"""Claim extraction: sections never overlap or cross a chapter; the code checks every claim of the model."""
from draft_coach.claims import (Claim, check_claim, extractable_segments, locate_quote, make_sections, names_in,
                                parent_chunk, stale_links)
from draft_coach.names import NameNormalizer

# one segment every 10 s, 5 words each; Piper is named at 20 s
SEGMENTS = [{"start_s": s, "text": f"w{s} a b c d", "mentions": []} for s in range(0, 100, 10)]
SEGMENTS[2] = {"start_s": 20, "text": "w20 Piper is great here", "mentions": ["piper"]}
MAPS = [{"id": "hot-potato", "name": "Hot Potato", "aliases": []}]
BRAWLERS = [{"id": i, "en": en, "it": en, "aliases": [], "context_aliases": []}
            for i, en in [("piper", "Piper"), ("nani", "Nani"), ("colt", "Colt")]]
NORMALIZER = NameNormalizer(BRAWLERS)
INFO = {"piper": {"class": "sniper", "wiki_class": "Marksman"}, "nani": {"class": "sniper", "wiki_class": None},
        "colt": {"class": "anti_tank", "wiki_class": "Damage Dealer"}}


def claim(**fields) -> Claim:
    base = dict(quote="Piper is great here", concept="Piper is strong on this map.", reason="", example="",
                topic="meta", brawlers=["Piper"], classes=[], effects=[], stats=[], relation="none",
                against_brawlers=[], against_classes=[], against_effects=[], draft_phase=[], rank_band=[], modes=[], maps=[], map_types=[], map_features=[], lane_role=[], mode_role=[])
    return Claim(**base | fields)


def check(c: Claim):
    return check_claim(c, make_sections(SEGMENTS, [])[0], MAPS, NORMALIZER, INFO)


def test_sections_cover_every_segment_once():
    sections = make_sections(SEGMENTS, [], max_words=12)
    assert [s["start_s"] for s in sections] == [0, 20, 40, 60, 80]
    assert sum(len(s["segments"]) for s in sections) == len(SEGMENTS)


def test_sections_never_cross_a_chapter():
    chapters = [{"start_s": 0, "title": "Intro"}, {"start_s": 30, "title": "1st Picks"}]
    sections = make_sections(SEGMENTS, chapters, max_words=100)
    assert [(s["chapter"], s["start_s"], s["end_s"]) for s in sections] == [("Intro", 0, 30), ("1st Picks", 30, 90)]


def test_screen_list_with_the_same_time_keeps_its_place():
    screen = SEGMENTS[:3] + [{"start_s": 20, "text": "[On screen] list", "mentions": []}] + SEGMENTS[3:]
    sections = make_sections(screen, [], max_words=12)  # 0-10 | 20 + screen list | 30-40 ...
    assert sections[1]["text"] == "w20 Piper is great here [On screen] list" and sections[1]["end_s"] == 30


def test_excluded_format_after_a_change_is_dropped():
    source = {"format": "guide", "format_changes": [{"start_s": 50, "format": "gameplay"}]}
    assert [s["start_s"] for s in extractable_segments(source, SEGMENTS)] == [0, 10, 20, 30, 40]


def test_quote_gives_the_interval_of_its_segments():
    section = make_sections(SEGMENTS, [])[0]
    assert locate_quote("PIPER is great, here... w30", section) == (20, 40)  # case and punctuation ignored


def test_quote_with_skipped_words_is_rejected():
    assert check(claim(quote="Piper is ... here")) == (None, "quote_not_found")


def test_brawler_the_speaker_did_not_say_is_rejected():
    assert check(claim(concept="Nani is strong on this map.")) == (None, "brawler_not_in_section")  # Nani for Najia


def test_brawler_named_only_in_the_chapter_title_is_kept():
    section = make_sections(SEGMENTS, [{"start_s": 0, "title": "Countering Colt"}])[0]
    record, reason = check_claim(claim(concept="Piper counters Colt.", relation="strong_against",
                                       against_brawlers=["Colt"]), section, MAPS, NORMALIZER, INFO)
    assert reason is None and (record["brawlers"], record["against_brawlers"]) == (["piper"], ["colt"])


def test_brawler_in_the_example_must_be_in_the_section():
    assert check(claim(example="Colt on Hot Potato")) == (None, "brawler_not_in_section")


def test_brawlers_and_their_classes_come_from_the_tables():
    record, reason = check(claim(maps=["Hot Potato", "Nowhere"], modes=["heist", "duels"], brawlers=["Piper", "Pipr"]))
    assert reason is None
    assert (record["brawlers"], record["unknown_brawlers"]) == (["piper"], ["Pipr"])
    assert record["brawler_classes"] == {"draft": ["sniper"], "game": ["Marksman"]}
    assert (record["map_ids"], record["unknown_maps"], record["modes"]) == (["hot-potato"], ["Nowhere"], ["heist"])


def test_other_side_needs_a_direction():
    assert check(claim(against_effects=["stun"])) == (None, "against_without_relation")


def test_weakness_to_a_mechanic_is_kept_with_its_direction():
    record, reason = check(claim(relation="weak_against", against_effects=["stun", "knockback"]))
    assert reason is None
    assert (record["brawlers"], record["relation"], record["against_effects"]) == (["piper"], "weak_against", ["stun", "knockback"])


def test_brawler_in_lowercase_in_the_transcript_is_kept():
    section = make_sections([{"start_s": 0, "text": "just go colt here", "mentions": []}], [])[0]
    record, reason = check_claim(claim(quote="go colt here", concept="Colt is good here.", brawlers=["Colt"]),
                                 section, MAPS, NORMALIZER, INFO)
    assert reason is None and record["brawlers"] == ["colt"]


def test_map_named_after_a_brawler_is_not_a_brawler():
    section = make_sections([{"start_s": 0, "text": "on Belle's Rock Piper is great here", "mentions": ["piper"]}], [])[0]
    brawlers = BRAWLERS + [{"id": "belle", "en": "Belle", "it": "Belle", "aliases": [], "context_aliases": []}]
    info = INFO | {"belle": {"class": "sniper", "wiki_class": None}}
    record, reason = check_claim(claim(concept="Piper is great on Belle's Rock."), section, MAPS, NameNormalizer(brawlers), info)
    assert reason is None and record["mentions"] == ["piper"]


def test_plural_of_a_brawler_name_is_the_brawler():
    assert names_in("two pipers on that map", NORMALIZER, []) == ["piper"]


def test_mode_roles_are_mapped_to_the_nine_roles():
    record, reason = check(claim(mode_role=["Gem Carrier", "aggro", "mid brawler", "anti-tank", "sweeper"]))
    assert reason is None
    assert record["mode_role"] == ["gem_carrier", "pressure"] and record["lane_role"] == ["mid"]
    assert record["mode_role_raw"] == ["Gem Carrier", "aggro", "mid brawler", "anti-tank", "sweeper"]
    assert record["unknown_mode_roles"] == ["sweeper"]  # a class word is dropped, a new label is counted



def test_parent_chunk_is_the_one_that_overlaps_the_claim_most():
    chunks = [{"_id": "a", "start_s": 0, "end_s": 60}, {"_id": "b", "start_s": 45, "end_s": 105}]
    assert parent_chunk({"start_s": 50, "end_s": 100}, chunks) == "b"
    assert parent_chunk({"start_s": 500, "end_s": 510}, chunks) is None  # no chunk of the source covers it


def test_stale_links_are_the_points_whose_parent_chunk_changed():
    stored = {"p1": "src:window60-v1:0", "p2": "src:chapter60-plain-v1:45", "p3": None, "gone": "src:window60-v1:90"}
    parents = {"p1": "src:chapter60-plain-v1:0", "p2": "src:chapter60-plain-v1:45", "p3": "src:chapter60-plain-v1:90"}
    assert stale_links(stored, parents) == {"p1": "src:chapter60-plain-v1:0", "p3": "src:chapter60-plain-v1:90"}