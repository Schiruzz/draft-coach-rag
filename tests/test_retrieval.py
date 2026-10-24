"""Retrieval helpers that do not need the services."""
from draft_coach.retrieval import minute


def test_minute_format():
    assert minute(0) == "0:00" and minute(125) == "2:05" and minute(3723) == "62:03"