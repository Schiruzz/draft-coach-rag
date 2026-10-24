"""Vector index helpers: ids are stable, names keep models apart, hashes follow the text."""
from draft_coach.vectors import collection_name, point_id, text_hash


def test_point_id_is_stable():
    assert point_id("src:window60-v1:0") == point_id("src:window60-v1:0")
    assert point_id("src:window60-v1:0") != point_id("src:window60-v1:45")


def test_collection_name_has_method_and_model():
    assert collection_name("window60-v1", "gemini-embedding-2") == "chunks__window60-v1__gemini-embedding-2"


def test_text_hash_changes_with_text():
    assert text_hash("Crow") != text_hash("Crow ")