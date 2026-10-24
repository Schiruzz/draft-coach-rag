"""Find the chunks nearest to a question in the vector index (shared by search and answer)."""
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from draft_coach.chunks import METHOD
from draft_coach.config import EMBEDDING_MODEL, QDRANT_URL
from draft_coach.vectors import collection_name

TOP_K = 5


class Retriever:
    def __init__(self, genai_client: genai.Client, qdrant: QdrantClient | None = None):
        self.genai = genai_client
        self.qdrant = qdrant or QdrantClient(url=QDRANT_URL)
        self.collection = collection_name(METHOD, EMBEDDING_MODEL)

    def search(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Question -> top_k chunk payloads, nearest first, each with its similarity score."""
        config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")  # questions are queries, chunks are documents
        vector = self.genai.models.embed_content(model=EMBEDDING_MODEL, contents=question, config=config).embeddings[0].values
        hits = self.qdrant.query_points(self.collection, query=vector, limit=top_k, with_payload=True).points
        return [{**hit.payload, "score": hit.score} for hit in hits]


def minute(seconds: int) -> str:
    """125 -> '2:05'"""
    return f"{seconds // 60}:{seconds % 60:02d}"