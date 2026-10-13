"""Both databases must answer. In CI they run as service containers."""
from draft_coach.health import mongo_version, qdrant_version


def test_mongodb_answers():
    assert mongo_version()


def test_qdrant_answers():
    assert qdrant_version()