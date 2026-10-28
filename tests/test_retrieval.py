"""Retrieval helpers, and the source filter on an in-memory Qdrant (no services needed)."""
from qdrant_client import QdrantClient, models

from draft_coach.retrieval import Retriever, minute


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