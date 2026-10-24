"""Time windows: every segment is covered, windows overlap, the loop always ends."""
from draft_coach.chunks import make_windows

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