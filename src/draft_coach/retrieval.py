"""Find the chunks nearest to a question in the vector index (shared by search and answer)."""
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

from draft_coach.chunks import METHOD
from draft_coach.config import EMBEDDING_MODEL, QDRANT_URL
from draft_coach.vectors import collection_name

TOP_K = 5


class Retriever:
    def __init__(self, genai_client: genai.Client, qdrant: QdrantClient | None = None, method: str = METHOD):
        self.genai = genai_client
        self.qdrant = qdrant or QdrantClient(url=QDRANT_URL)
        self.collection = collection_name(method, EMBEDDING_MODEL)  # the chunking method picks the collection

    def search(self, question: str, top_k: int = TOP_K, sources: list[str] | None = None) -> list[dict]:
        """Question -> top_k chunk payloads, nearest first, each with its similarity score.
        sources: search only the chunks of these source ids (None = all)."""
        config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")  # questions are queries, chunks are documents
        vector = self.genai.models.embed_content(model=EMBEDDING_MODEL, contents=question, config=config).embeddings[0].values
        only = None if sources is None else models.Filter(
            must=[models.FieldCondition(key="source_id", match=models.MatchAny(any=sources))])
        hits = self.qdrant.query_points(self.collection, query=vector, query_filter=only,
                                        limit=top_k, with_payload=True).points
        return [{**hit.payload, "score": hit.score} for hit in hits]


def minute(seconds: int) -> str:
    """125 -> '2:05'"""
    return f"{seconds // 60}:{seconds % 60:02d}"