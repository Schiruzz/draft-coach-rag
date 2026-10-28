"""Retrieval helpers, and the source filter on an in-memory Qdrant (no services needed)."""
from qdrant_client import QdrantClient, models

from draft_coach.retrieval import Retriever, locator, minute
from draft_coach.wiki_chunks import LAYER as WIKI_LAYER


def test_minute_format():
    assert minute(0) == "0:00" and minute(125) == "2:05" and minute(3723) == "62:03"


class FakeGenai:  # every text gets the same vector: only the filter decides what comes back
    class models:
        @staticmethod
        def embed_content(**kwargs):
            return type("R", (), {"embeddings": [type("E", (), {"values": [1.0, 0.0]})]})


def test_search_keeps_only_the_given_sources():
    qdrant = QdrantClient(":memory:")
    retriever = Retriever(FakeGenai(), qdrant)
    qdrant.create_collection(retriever.collection, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    qdrant.upsert(retriever.collection, [models.PointStruct(id=i, vector=[1.0, 0.0], payload={"source_id": s})
                                         for i, s in enumerate(["guide-a", "gameplay-b", "guide-c"])])
    found = {c["source_id"] for c in retriever.search("q", top_k=5, sources=["guide-a", "guide-c"])}
    assert found == {"guide-a", "guide-c"}
    assert len(retriever.search("q", top_k=5)) == 3  # no filter: everything


def test_locator_minute_or_section():
    assert locator({"start_s": 125}) == "2:05" and locator({"section": "Tips 2"}) == "Tips 2"


def test_layers_never_mix():
    qdrant = QdrantClient(":memory:")
    retriever = Retriever(FakeGenai(), qdrant)
    qdrant.create_collection(retriever.collection, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    qdrant.upsert(retriever.collection, [
        models.PointStruct(id=0, vector=[1.0, 0.0], payload={"source_id": "guide-a"}),
        models.PointStruct(id=1, vector=[1.0, 0.0], payload={"source_id": "fandom-heist", "layer": WIKI_LAYER}),
        models.PointStruct(id=2, vector=[1.0, 0.0], payload={"source_id": "fandom-map-hot-potato", "layer": WIKI_LAYER})])
    assert {c["source_id"] for c in retriever.search("q", top_k=5)} == {"guide-a"}  # default: the guides only
    assert {c["source_id"] for c in retriever.search("q", top_k=5, layer=WIKI_LAYER)} == {"fandom-heist", "fandom-map-hot-potato"}
    assert [c["source_id"] for c in retriever.search("q", top_k=5, sources=["fandom-heist"], layer=WIKI_LAYER)] == ["fandom-heist"]