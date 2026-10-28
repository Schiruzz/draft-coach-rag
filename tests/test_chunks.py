"""Time windows: every segment is covered, windows overlap, the loop always ends."""
from draft_coach.chunks import make_chapter_windows, make_windows

# one segment every 10 s for 3 minutes; Piper is named at 50 s
SEGMENTS = [{"start_s": s, "text": f"t{s}", "mentions": ["piper"] if s == 50 else []} for s in range(0, 180, 10)]


def test_first_window_covers_60_seconds():
    first = make_windows(SEGMENTS)[0]
    assert (first["start_s"], first["end_s"]) == (0, 60)
    assert first["text"] == "t0 t10 t20 t30 t40 t50"


def test_next_window_starts_inside_the_overlap():
    assert [w["start_s"] for w in make_windows(SEGMENTS)] == [0, 50, 100, 150]


def test_every_segment_is_in_some_window():
    covered = {word for w in make_windows(SEGMENTS) for word in w["text"].split()}
    assert covered == {s["text"] for s in SEGMENTS}


def test_mentions_are_merged():
    windows = make_windows(SEGMENTS)
    assert windows[0]["mentions"] == ["piper"] and windows[1]["mentions"] == ["piper"]


def test_long_segment_gap_still_moves_forward():
    gappy = [{"start_s": s, "text": str(s), "mentions": []} for s in (0, 200, 205)]
    assert [w["start_s"] for w in make_windows(gappy)] == [0, 200]


# two chapters on the same 3 minutes: the second one starts at 90 s
CHAPTERS = [{"start_s": 0, "title": "Intro"}, {"start_s": 90, "title": "Heist - Hot Potato"}]


def test_chapter_windows_never_cross_a_chapter():
    windows = make_chapter_windows(SEGMENTS, CHAPTERS)
    for w in windows:
        assert (w["start_s"] < 90) == (w["chapter"] == "Intro")
        assert w["end_s"] <= 90 or w["start_s"] >= 90


def test_chapter_title_starts_the_text():
    windows = make_chapter_windows(SEGMENTS, CHAPTERS)
    assert windows[0]["text"].startswith("Intro: t0")
    assert windows[-1]["text"].startswith("Heist - Hot Potato: ")


def test_last_window_of_a_chapter_ends_at_the_next_chapter():
    intro = [w for w in make_chapter_windows(SEGMENTS, CHAPTERS) if w["chapter"] == "Intro"]
    assert intro[-1]["end_s"] == 90


def test_every_segment_is_in_some_chapter_window():
    covered = {word for w in make_chapter_windows(SEGMENTS, CHAPTERS) for word in w["text"].split()}
    assert {s["text"] for s in SEGMENTS} <= covered