"""Transcript parsing: timestamps become seconds, text keeps its order."""
from draft_coach.transcript import parse_transcript

SAMPLE = "Intro line\n0:06\nsecond part\ncontinues here\n1:02:03\nlast part\n"


def test_first_segment_starts_at_zero():
    assert parse_transcript(SAMPLE)[0] == {"start_s": 0, "text": "Intro line"}


def test_timestamps_become_seconds():
    assert [s["start_s"] for s in parse_transcript(SAMPLE)] == [0, 6, 3723]


def test_lines_between_timestamps_are_joined():
    assert parse_transcript(SAMPLE)[1]["text"] == "second part continues here"