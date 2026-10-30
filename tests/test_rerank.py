"""The reranker keeps the ranker's order and every payload field, and asks for the first 5 only."""
import pytest
from google.cloud import discoveryengine_v1 as discoveryengine

from draft_coach.rerank import TOP_N, Reranker


class FakeClient:
    """Ranks the records in reverse order, as a real ranker could; keeps the last request."""
    def rank(self, request, retry, timeout):
        self.request = request
        ids = [r.id for r in request.records][::-1][:request.top_n]
        return discoveryengine.RankResponse(records=[discoveryengine.RankingRecord(id=i, score=0.9) for i in ids])


CHUNKS = [{"chunk_id": f"src:m:{i}", "text": f"text {i}", "mentions": []} for i in range(7)]


def test_rerank_returns_the_ranker_order_with_the_payload():
    reranker = Reranker("my-project", "semantic-ranker-fast-005", client=FakeClient())
    top = reranker.rerank("who counters Bull?", CHUNKS)
    assert [c["chunk_id"] for c in top] == ["src:m:6", "src:m:5", "src:m:4", "src:m:3", "src:m:2"]
    assert top[0]["text"] == "text 6" and top[0]["rank_score"] == pytest.approx(0.9)  # the API returns float32


def test_rerank_request_carries_question_model_and_top_n():
    client = FakeClient()
    Reranker("my-project", "semantic-ranker-fast-005", client=client).rerank("who counters Bull?", CHUNKS)
    assert client.request.query == "who counters Bull?" and client.request.model == "semantic-ranker-fast-005"
    assert client.request.top_n == TOP_N and len(client.request.records) == len(CHUNKS)
    assert client.request.ranking_config == "projects/my-project/locations/global/rankingConfigs/default_ranking_config"