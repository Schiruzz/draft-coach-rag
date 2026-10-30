"""Step 7b: reorder the chunks found by similarity with the Vertex AI ranking API.
A cross-encoder reads the question and each chunk together: slower than comparing two vectors, more precise.
The ranking API is served only in the location `global` (Gemini stays in `eu`)."""
from google.api_core import exceptions, retry
from google.cloud import discoveryengine_v1 as discoveryengine

RANK_MODELS = ["semantic-ranker-default-005", "semantic-ranker-fast-005"]  # pinned versions: reproducible results
RANK_MODEL = "semantic-ranker-default-005"  # chosen 30/10: more useful chunks than fast, latency irrelevant (0.09 s)
TOP_N = 5          # the answer reads 5 chunks
TIMEOUT_S = 2      # measured 30/10: p95 0.21 s (default), 0.10 s (fast), max 0.34 s; 2 s leaves room for a slow call
TEMPORARY = retry.if_exception_type(exceptions.TooManyRequests, exceptions.ServiceUnavailable,
                                    exceptions.InternalServerError, exceptions.DeadlineExceeded)


class Reranker:
    def __init__(self, project: str, model: str, client: discoveryengine.RankServiceClient | None = None):
        self.client = client or discoveryengine.RankServiceClient()  # credentials from GOOGLE_APPLICATION_CREDENTIALS
        self.config = discoveryengine.RankServiceClient.ranking_config_path(project, "global", "default_ranking_config")
        self.model = model
        # temporary errors: retried after 1, 2, 4... s, one line each, at most 4 timeouts in all
        self.retry = retry.Retry(predicate=TEMPORARY, initial=1, multiplier=2, maximum=16, timeout=4 * TIMEOUT_S,
                                 on_error=lambda error: print(f"rank {model}: {type(error).__name__}, retrying"))

    def rerank(self, question: str, chunks: list[dict]) -> list[dict]:
        """The TOP_N chunks in the ranker's order, each with its rank_score; every payload field is kept."""
        records = [discoveryengine.RankingRecord(id=str(i), content=c["text"]) for i, c in enumerate(chunks)]
        request = discoveryengine.RankRequest(ranking_config=self.config, model=self.model, query=question,
                                              records=records, top_n=TOP_N, ignore_record_details_in_response=True)
        response = self.client.rank(request=request, retry=self.retry, timeout=TIMEOUT_S)
        return [{**chunks[int(r.id)], "rank_score": r.score} for r in response.records]