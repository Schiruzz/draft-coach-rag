"""Find the chunks for a question: nearest in the vector index, then reordered by the ranker (shared by search and answer)."""
from google import genai
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.genai import types
from qdrant_client import QdrantClient, models

from draft_coach.chunks import METHOD
from draft_coach.config import EMBEDDING_MODEL, GOOGLE_CLOUD_PROJECT, QDRANT_URL
from draft_coach.glossary import load_terms, to_english
from draft_coach.rerank import RANK_MODEL, Reranker
from draft_coach.vectors import collection_name

TOP_K = 5
CANDIDATES = 20  # measured 30/10: J@20 97% on written questions; the ranker picks the 5 best of these


class Retriever:
    def __init__(self, genai_client: genai.Client, qdrant: QdrantClient | None = None, method: str = METHOD,
                 reranker: Reranker | None = None):
        self.genai = genai_client
        self.qdrant = qdrant or QdrantClient(url=QDRANT_URL)
        self.reranker = reranker  # created on the first retrieve(): search() alone needs no ranking credentials
        self.collection = collection_name(method, EMBEDDING_MODEL)  # the chunking method picks the collection
        self.terms = load_terms()  # Italian game terms -> English terms of the corpus (step 7d)

    def search(self, question: str, top_k: int = TOP_K, sources: list[str] | None = None,
               layer: str | None = None) -> list[dict]:
        """Question -> top_k chunk payloads, nearest first, each with its similarity score.
        sources: search only the chunks of these source ids (None = all).
        layer: None = the guides (transcript chunks, no layer field), "wiki" = the wiki layer (step 5b).
        The two layers never compete for the same places."""
        config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")  # questions are queries, chunks are documents
        vector = self.genai.models.embed_content(model=EMBEDDING_MODEL, contents=question, config=config).embeddings[0].values
        must = [models.IsEmptyCondition(is_empty=models.PayloadField(key="layer"))] if layer is None else \
               [models.FieldCondition(key="layer", match=models.MatchValue(value=layer))]
        if sources is not None:
            must.append(models.FieldCondition(key="source_id", match=models.MatchAny(any=sources)))
        hits = self.qdrant.query_points(self.collection, query=vector, query_filter=models.Filter(must=must),
                                        limit=top_k, with_payload=True).points
        return [{**hit.payload, "score": hit.score} for hit in hits]


    def retrieve(self, question: str, sources: list[str] | None = None) -> list[dict]:
        """The retrieval in use (30/10): the CANDIDATES nearest guide chunks, reordered by the ranker, first TOP_K.
        If the ranker fails, the TOP_K nearest: an answer from similarity alone beats no answer.
        Italian game terms are replaced with the English ones first (measured 31/10, scripts/measure_glossary.py)."""
        question = to_english(question, self.terms)
        nearest = self.search(question, top_k=CANDIDATES, sources=sources)
        if self.reranker is None:
            self.reranker = Reranker(GOOGLE_CLOUD_PROJECT, RANK_MODEL)
        try:
            return self.reranker.rerank(question, nearest)
        except (GoogleAPICallError, RetryError) as error:
            print(f"rerank failed ({type(error).__name__}): similarity order kept")
            return nearest[:TOP_K]


def minute(seconds: int) -> str:
    """125 -> '2:05'"""
    return f"{seconds // 60}:{seconds % 60:02d}"


def locator(chunk: dict) -> str:
    """Where the chunk is in its source: the minute for a video, the section for a wiki page."""
    return minute(chunk["start_s"]) if "start_s" in chunk else chunk["section"]