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


def minute(seconds: int) -> str:
    """125 -> '2:05'"""
    return f"{seconds // 60}:{seconds % 60:02d}"


def locator(chunk: dict) -> str:
    """Where the chunk is in its source: the minute for a video, the section for a wiki page."""
    return minute(chunk["start_s"]) if "start_s" in chunk else chunk["section"]